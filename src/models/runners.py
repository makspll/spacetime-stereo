from random import randint
import time
import torch.nn.functional as F
from torch.autograd.variable import Variable
from .augmentations.image_prep import kitti_transform
import os 
import torch 
import sys
from .metrics import bad_n_error, AreaSource

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPRODS_PATH = os.path.join(SCRIPT_DIR,'..','reproductions')

class OpticalFlowWarpGT():
    def __init__(self) -> None:
        pass


class GenericRunner():
    def __init__(self, model_cls, args, training=False) -> None:
        self.keys = set()
        self.model_cls = model_cls

    def get_keys(self):
        return self.keys
    
    def get_output(self, model, sample, keys):

        inputs,targets = self.get_model_io_from_sample(sample, keys)

        start_time = time.time()
        with torch.no_grad():
            prediction = model(*[Variable(torch.tensor(x),requires_grad=False).cuda() for x in inputs])
        end_time = time.time()
        
        temp = prediction.cpu()
        temp = temp.detach()
        temp = temp.numpy()

        output_resolution = sample[keys['resolution']]
        height = temp.shape[-2]
        width = temp.shape[-1]

        if output_resolution[0] <= height and output_resolution[1]<= width:
            temp = temp[:, height - output_resolution[0]: height, width - output_resolution[1]: width]

        temp = temp[0, :, :]
        return {
            "runtime" : end_time - start_time,
            "outputs" : temp,
        }

    def get_model(self, weights_path, weights_source):
        torch.backends.cudnn.benchmark = True

        cuda = self.device == 'cuda'
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        model = self.model_cls()

        if cuda:
            model = torch.nn.DataParallel(model).cuda()
        
        if weights_path: #opt.resume:
            if os.path.isfile(weights_path):
                print("=> loading checkpoint '{}'".format(weights_path))
                checkpoint = torch.load(weights_path)
                model.load_state_dict(model.module.convert_weights(checkpoint['state_dict'],weights_source), strict=True)      
            else:
                print("=> no checkpoint found at '{}'".format(weights_path))
        return model

    def get_model_io(self,batch):
        raise NotImplementedError()

    def get_model_io_from_sample(self, sample, keys):
        raise NotImplementedError()

    def transform(self, inputs, keys, is_test_phase):
        raise NotImplementedError()

    def loss_accuracy_function(self, outputs, targets):
        raise NotImplementedError()

    def validate(self, model, loader):
        valid_iteration = 0
        epoch_loss = 0
        acc_all = 0
        model.eval()
        for iteration, batch in enumerate(loader):

            inputs,targets = self.get_model_io(batch)
            inputs = [Variable(x,requires_grad=False) for x in inputs]

            if self.device == 'cuda':
                inputs =  [x.cuda() for x in inputs]
                targets = [x.cuda() for x in targets]

            with torch.no_grad(): 
                outputs = model(*inputs)
                (loss,acc) = self.loss_accuracy_function(outputs,targets)

                epoch_loss += loss.item()
                acc_all += acc
                valid_iteration += 1

                print("===> Test({}/{}): Accuracy: ({:.4f})".format(iteration, len(loader), acc))
                sys.stdout.flush()

        print("===> Test: Avg. Accuracy: ({:.4f})".format(acc_all/valid_iteration))
        print(type(acc_all),type(valid_iteration),type(epoch_loss))
        return (acc_all/valid_iteration, epoch_loss /valid_iteration)


    def train(self, epoch, model, loader, optimizer):
        epoch_loss = 0
        valid_iteration = 0
        acc_all = 0

        for iteration, batch in enumerate(loader):

            inputs,targets = self.get_model_io(batch)
            inputs = [Variable(x,requires_grad=False) for x in inputs]
            
            if self.device =='cuda':
                inputs = [x.cuda() for x in inputs]
                targets = [x.cuda() for x in targets]

            train_start_time = time.time()
            model.train()
    
            optimizer.zero_grad()
            outputs = model(*inputs) 
            loss,acc = self.loss_accuracy_function(outputs,targets)

            if(loss):
                loss.backward()
                epoch_loss += loss.item()
                valid_iteration += 1
                optimizer.step()
                acc_all += acc
                train_end_time = time.time()
                train_time = train_end_time - train_start_time
                print("===> Epoch[{}]({}/{}): Loss: ({:.4f}), Acc.: ({:.4f}) Time: ({:.2f}s)".format(epoch, iteration, len(loader), loss.item(),acc,train_time))
            else:
                print(f"===> Epoch[{epoch}]: Skipped " )

            sys.stdout.flush()
                                    
        print("===> Epoch {} Complete: Avg. Loss: ({:.4f}), Avg. Acc.: ({:.4f})".format(epoch, epoch_loss / valid_iteration, acc_all/ valid_iteration))
        return (acc_all / valid_iteration, (epoch_loss/ valid_iteration))

class LEASTereoRunner(GenericRunner):
    
    def __init__(self,args, training=False) -> None:
        from models.LEAStereo import LEAStereo

        super().__init__(LEAStereo,args,training)
        self.args = args
        
        dataset = args.dataset
        self.crop_width = 1248
        self.crop_height = 384 
        self.device = 'cuda'
        self.maxdisp = 192
        self.training = training 
        self.last_crop = None
        if dataset == 'sceneflow':
            self.crop_width = 960
            self.crop_height = 576
        if self.training:
            self.keys = set(['l0','r0','d0'])
        else:
            self.keys = (['l0','r0','l1','r1','d0','d0noc','d1','d1noc','fgmap','resolution','index'])
    
    def transform(self, inputs, keys, is_test_phase):

        h,w,c = inputs[keys['l0']].shape[-3:]
        new_height = self.crop_height
        new_width = self.crop_width
        random_crop = None
        # left_right_rand = None
        if self.training and not is_test_phase:
            new_height = 168 #high_res 288 # low_res 168
            new_width = 336 #high_res 576 # low_res 336
            random_crop = (randint(0,w-new_width),
                        randint(0,h-new_height))
            # left_right_rand = randint(0,1) == 1


        inputs[keys['l0']] = kitti_transform(inputs[keys['l0']], new_height, new_width, start_corner=random_crop) 
        inputs[keys['r0']] = kitti_transform(inputs[keys['r0']], new_height, new_width, start_corner=random_crop) 
        inputs[keys['d0']] = kitti_transform(inputs[keys['d0']], new_height, new_width, start_corner=random_crop,normalize_rgb=False) 
        if not self.training:
            inputs[keys['d0noc']] = kitti_transform(inputs[keys['d0noc']], new_height, new_width, start_corner=random_crop,normalize_rgb=False) 
            inputs[keys['d1noc']] = kitti_transform(inputs[keys['d1noc']], new_height, new_width, start_corner=random_crop,normalize_rgb=False) 
            inputs[keys['d1']] = kitti_transform(inputs[keys['d1']], new_height, new_width, start_corner=random_crop,normalize_rgb=False) 
            inputs[keys['fgmap']] = kitti_transform(inputs[keys['fgmap']], new_height, new_width, start_corner=random_crop,normalize_rgb=False) 

        return inputs 


    def get_model_io_from_sample(self, sample, keys):
        return [sample[keys['l0']],sample[keys['r0']]], [sample[keys['d0']]]

    def get_model_io(self,batch):
        return [torch.squeeze(batch[0],1),torch.squeeze(batch[1],1)], [torch.squeeze(batch[2],1).float()]

    def loss_accuracy_function(self, outputs, targets):
            target= targets[0]
            output = outputs
            mask = (target < self.maxdisp) & (target > 0)
            mask.detach_()
            valid_target_px = target[mask]

            acc = 100 - bad_n_error(3,outputs.detach().cpu().numpy(),target.detach().cpu().numpy(),AreaSource.BOTH,max_disp=self.maxdisp)
            if(valid_target_px.size()[0] <= 0):
                return (0,acc)
            else:
                return (F.smooth_l1_loss(output[mask],valid_target_px),acc)

class STSEarlyFusionConcatRunner(LEASTereoRunner):
    def __init__(self,args, training=False) -> None:
        from models.STSEarlyFusionConcat import STSEarlyFusionConcat

        super().__init__(args,training)
        self.model_cls = STSEarlyFusionConcat

        if self.training:
            self.keys = set(['l0','r0','l1','r1','d0','d1'])
        else:
            self.keys = (['l0','r0','l1','r1','d0','d0noc','d1','d1noc','fgmap','resolution','index'])

    def transform(self, inputs, keys, is_test_phase):

        h,w,c = inputs[keys['l0']].shape[-3:]
        new_height = self.crop_height
        new_width = self.crop_width
        random_crop = None
        # left_right_rand = None
        if self.training and not is_test_phase:
            new_height = 168 #high_res 288 # low_res 168
            new_width = 336 #high_res 576 # low_res 336
            random_crop = (randint(0,w-new_width),
                        randint(0,h-new_height))
            # left_right_rand = randint(0,1) == 1


        inputs[keys['l0']] = kitti_transform(inputs[keys['l0']], new_height, new_width, start_corner=random_crop) 
        inputs[keys['r0']] = kitti_transform(inputs[keys['r0']], new_height, new_width, start_corner=random_crop) 
        inputs[keys['l1']] = kitti_transform(inputs[keys['l1']], new_height, new_width, start_corner=random_crop) 
        inputs[keys['r1']] = kitti_transform(inputs[keys['r1']], new_height, new_width, start_corner=random_crop) 
        inputs[keys['d0']] = kitti_transform(inputs[keys['d0']], new_height, new_width, start_corner=random_crop,normalize_rgb=False) 
        inputs[keys['d1']] = kitti_transform(inputs[keys['d1']], new_height, new_width, start_corner=random_crop,normalize_rgb=False) 
        
        if not self.training:
            inputs[keys['d0noc']] = kitti_transform(inputs[keys['d0noc']], new_height, new_width, start_corner=random_crop,normalize_rgb=False) 
            inputs[keys['d1noc']] = kitti_transform(inputs[keys['d1noc']], new_height, new_width, start_corner=random_crop,normalize_rgb=False) 
            inputs[keys['d1']] = kitti_transform(inputs[keys['d1']], new_height, new_width, start_corner=random_crop,normalize_rgb=False) 
            inputs[keys['fgmap']] = kitti_transform(inputs[keys['fgmap']], new_height, new_width, start_corner=random_crop,normalize_rgb=False) 

        return inputs 

    def get_model_io_from_sample(self, sample, keys):
        return [sample[keys['l0']],sample[keys['r0']],sample[keys['l1']],sample[keys['r1']]], [sample[keys['d0']],sample[keys['d1']]]

    def get_model_io(self,batch):
        return [torch.squeeze(batch[0],1),
                torch.squeeze(batch[1],1),
                torch.squeeze(batch[2],1),
                torch.squeeze(batch[3],1)], [torch.squeeze(batch[4],1).float(),
                                            torch.squeeze(batch[5],1).float()]


    # def loss_accuracy_function(self, outputs, targets):
    #         target = targets[0]
    #         output = outputs
    #         mask = (target < self.maxdisp) & (target > 0)
    #         mask.detach_()
    #         valid_target_px = target[mask]

    #         acc = 100 - bad_n_error(3,outputs.detach().cpu().numpy(),target.detach().cpu().numpy(),AreaSource.BOTH,max_disp=self.maxdisp)
    #         if(valid_target_px.size()[0] <= 0):
    #             return (0,acc)
    #         else:
    #             return (F.smooth_l1_loss(output[mask],valid_target_px),acc)