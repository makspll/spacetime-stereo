from random import randint
import time
import torch.nn.functional as F

from torch.autograd.variable import Variable

from .augmentations.image_prep import kitti_transform
from fetch_network import get_leastereo
import os 
from skimage import io
import numpy as np
import torch 
import sys
from .metrics import bad_n_error, AreaSource

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPRODS_PATH = os.path.join(SCRIPT_DIR,'..','reproductions')

class OpticalFlowWarpGT():
    def __init__(self) -> None:
        pass


class GenericRunner():
    def __init__(self, args, training=False) -> None:
        pass 
    
    def get_keys():
        raise NotImplementedError()
    
    def get_output(self, model, sample, keys):
        raise NotImplementedError()

    def get_model(self, weights_path):
        raise NotImplementedError()

    def validate(self, model,loader):
        valid_iteration = 0
        epoch_loss = 0
        three_px_acc_all = 0
        model.eval()
        for iteration, batch in enumerate(loader):
            input1, input2, target = [Variable(x,requires_grad=False) for x in batch]
            if self.device == 'cuda':
                input1 = input1.cuda()
                input2 = input2.cuda()
                target = target.cuda()

            input1 = torch.squeeze(input1,1)
            input2 = torch.squeeze(input2,1)
            target=torch.squeeze(target,1)
            mask = (target < self.maxdisp) & (target > 0)
            mask.detach_()
            valid=target[mask].size()[0]
            if valid>0:
                with torch.no_grad(): 
                    disp = model(input1,input2)
                    loss = F.smooth_l1_loss(disp[mask], target[mask], reduction='mean')
                    loss = loss.cpu().detach().numpy()
                    epoch_loss += loss
                    valid_iteration += 1

                    pred_disp = disp.cpu().detach().numpy()
                    true_disp = target.cpu().detach().numpy()
                    three_px_acc = 100-bad_n_error(3,pred_disp,true_disp,AreaSource.BOTH,max_disp=self.maxdisp)
                    three_px_acc_all += three_px_acc
        
                    print("===> Test({}/{}): Accuracy: ({:.4f})".format(iteration, len(loader), three_px_acc))
                    sys.stdout.flush()

        print("===> Test: Avg. Accuracy: ({:.4f})".format(three_px_acc_all/valid_iteration))
        return (three_px_acc_all/valid_iteration, epoch_loss /valid_iteration)


    def train(self, epoch, model, loader, optimizer):
        epoch_loss = 0
        valid_iteration = 0
        three_px_acc_all = 0

        for iteration, batch in enumerate(loader):

            input1, input2, target = [Variable(x,requires_grad=False) for x in batch]
            
            if self.device =='cuda':
                input1 = input1.cuda()
                input2 = input2.cuda()
                target = target.cuda().float()
            input1 = torch.squeeze(input1,1)
            input2 = torch.squeeze(input2,1)

            target=torch.squeeze(target,1)
            mask = (target < self.maxdisp) & (target > 0)
            mask.detach_()
            valid = target[mask].size()[0]

            train_start_time = time.time()
            if valid > 0:
                model.train()
        
                optimizer.zero_grad()
                disp = model(input1,input2) 

                loss = F.smooth_l1_loss(disp[mask], target[mask], reduction='mean')
                loss.backward()

                optimizer.step()

                
                disp = disp.cpu().detach().numpy()
                target = target.cpu().detach().numpy()

                three_px_acc = 100-bad_n_error(3,disp,target,max_disp=self.maxdisp)
                three_px_acc_all += three_px_acc

                train_end_time = time.time()
                train_time = train_end_time - train_start_time

                epoch_loss += loss.item()
                valid_iteration += 1
                print("===> Epoch[{}]({}/{}): Loss: ({:.4f}), Acc.: ({:.4f}) Time: ({:.2f}s)".format(epoch, iteration, len(loader), loss.item(),three_px_acc,train_time))
                sys.stdout.flush()
                                    
        print("===> Epoch {} Complete: Avg. Loss: ({:.4f}), Avg. Acc.: ({:.4f})".format(epoch, epoch_loss / valid_iteration, three_px_acc_all/ valid_iteration))
        return (three_px_acc_all / valid_iteration, (epoch_loss/ valid_iteration))

class LEASTereoRunner():
    def __init__(self,args, training=False) -> None:
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
            new_height = 288#168
            new_width = 576#336
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
        
    def get_output(self, model, sample, keys):

        start_time = time.time()
        with torch.no_grad():
            prediction = model(Variable(torch.tensor(sample[keys['l0']]),requires_grad=False).cuda(),
                                Variable(torch.tensor(sample[keys['r0']]),requires_grad=False).cuda())
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

        

    def get_model(self, weights_path):

        return get_leastereo(weights_path)

    def get_keys(self):
        return self.keys 

    def validate(self, model,loader):
        valid_iteration = 0
        epoch_loss = 0
        three_px_acc_all = 0
        model.eval()
        for iteration, batch in enumerate(loader):
            input1, input2, target = [Variable(x,requires_grad=False) for x in batch]
            if self.device == 'cuda':
                input1 = input1.cuda()
                input2 = input2.cuda()
                target = target.cuda()

            input1 = torch.squeeze(input1,1)
            input2 = torch.squeeze(input2,1)
            target=torch.squeeze(target,1)
            mask = (target < self.maxdisp) & (target > 0)
            mask.detach_()
            valid=target[mask].size()[0]
            if valid>0:
                with torch.no_grad(): 
                    disp = model(input1,input2)
                    loss = F.smooth_l1_loss(disp[mask], target[mask], reduction='mean')
                    loss = loss.cpu().detach().numpy()
                    epoch_loss += loss
                    valid_iteration += 1

                    pred_disp = disp.cpu().detach().numpy()
                    true_disp = target.cpu().detach().numpy()
                    three_px_acc = 100-bad_n_error(3,pred_disp,true_disp,AreaSource.BOTH,max_disp=self.maxdisp)
                    three_px_acc_all += three_px_acc
        
                    print("===> Test({}/{}): Accuracy: ({:.4f})".format(iteration, len(loader), three_px_acc))
                    sys.stdout.flush()

        print("===> Test: Avg. Accuracy: ({:.4f})".format(three_px_acc_all/valid_iteration))
        return (three_px_acc_all/valid_iteration, epoch_loss /valid_iteration)


    def train(self, epoch, model, loader, optimizer):
        epoch_loss = 0
        valid_iteration = 0
        three_px_acc_all = 0

        for iteration, batch in enumerate(loader):

            input1, input2, target = [Variable(x,requires_grad=False) for x in batch]
            
            if self.device =='cuda':
                input1 = input1.cuda()
                input2 = input2.cuda()
                target = target.cuda().float()
            input1 = torch.squeeze(input1,1)
            input2 = torch.squeeze(input2,1)

            target=torch.squeeze(target,1)
            mask = (target < self.maxdisp) & (target > 0)
            mask.detach_()
            valid = target[mask].size()[0]

            train_start_time = time.time()
            if valid > 0:
                model.train()
        
                optimizer.zero_grad()
                disp = model(input1,input2) 

                loss = F.smooth_l1_loss(disp[mask], target[mask], reduction='mean')
                loss.backward()

                optimizer.step()

                
                disp = disp.cpu().detach().numpy()
                target = target.cpu().detach().numpy()

                three_px_acc = 100-bad_n_error(3,disp,target,max_disp=self.maxdisp)
                three_px_acc_all += three_px_acc

                train_end_time = time.time()
                train_time = train_end_time - train_start_time

                epoch_loss += loss.item()
                valid_iteration += 1
                print("===> Epoch[{}]({}/{}): Loss: ({:.4f}), Acc.: ({:.4f}) Time: ({:.2f}s)".format(epoch, iteration, len(loader), loss.item(),three_px_acc,train_time))
                sys.stdout.flush()
                                    
        print("===> Epoch {} Complete: Avg. Loss: ({:.4f}), Avg. Acc.: ({:.4f})".format(epoch, epoch_loss / valid_iteration, three_px_acc_all/ valid_iteration))
        return (three_px_acc_all / valid_iteration, (epoch_loss/ valid_iteration))

class STSEarlyFusionConcatRunner(LEASTereoRunner):
    pass