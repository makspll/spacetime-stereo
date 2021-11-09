import json
import os
from time import time
import numpy as np
import torch
from torch import optim
from models.runners import LEASTereoRunner, STSEarlyFusionConcatRunner 
from datasets import Kitti15Dataset
from args import PARSER_TRAIN
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator

torch.manual_seed(0)
import random
random.seed(0)
np.random.seed(0)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def get_splits(path):
    with open(path,'r') as f:
        return json.load(f)



METHODS = {
    'LEAStereo': lambda args: LEASTereoRunner(args,training=True),
    'STSEarlyFusionConcat': lambda args: STSEarlyFusionConcatRunner(args,training=True)
}
DATASETS = {
    'kitti2015': lambda *args: Kitti15Dataset(os.path.join(SCRIPT_DIR,'..','datasets','kitti2015'),*args)
}


def save_architecture(path,epoch,model,optimizer, scheduler, best, **kwargs):
    filename = os.path.join(path,"epoch_{}.pth".format(epoch) if not best else "best.pth")

    torch.save({
        'epoch' : epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        **kwargs
    }, filename)

    print("Checkpoint saved to {}".format(filename))

def make_plot(path, epoch, accuracies_train, losses_train, accuracies_val, losses_val, start_epoch = 1):

    with plt.style.context('ggplot'):
        fig = plt.figure(1,figsize=(8,6))
        ticks = np.arange(start_epoch,epoch+1)
        plt.plot(ticks, accuracies_train , label='Acc. (Train)')
        plt.plot(ticks, accuracies_val , label='Acc. (Val)')
        plt.plot(ticks, losses_train , label='Loss (Train)',linestyle='dashed')
        plt.plot(ticks, losses_val , label='Loss (Val)',linestyle='dashed')

        ax = plt.gca()
        ax.set_ylim(0,1)
        ax.set_xlim(start_epoch,epoch)
        ax.set_xlabel('Epoch')
        ax.xaxis.set_major_locator(MaxNLocator(integer=1))
        # For the minor ticks, use no labels; default NullFormatter.
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.legend()
        fig.savefig(os.path.join(path,'training_graph.pdf'))
        plt.clf()

if __name__ == "__main__":

    args = PARSER_TRAIN.parse_args()
    print(f"===> Building model with parameters: \n{args}\n")

    splits = get_splits(args.file)
    method = METHODS[args.method](args)
    epochs = args.epochs
    batch_size = args.batch

    indices_val = splits[args.dataset][args.method]["training"][args.valsplit]

    dataset_val = DATASETS[args.dataset](
        True, 
        indices_val,
        method.transform,
        True, # test phase
        method.get_keys())

    indices_train = [splits[args.dataset][args.method]["training"][x] for x in args.trainsplits]
    dataset_train = DATASETS[args.dataset](
        True, 
        [item for sublist in indices_train for item in sublist], # flatten 
        method.transform,
        False, # no test phase
        method.get_keys())

    resume_path = args.resume
    model = method.get_model(resume_path)
    model.eval()

    torch.backends.cudnn.benchmark = True

    out_path = os.path.join(SCRIPT_DIR,args.save)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    train_loader = DataLoader(dataset_train, batch_size, shuffle=True,pin_memory=True)
    val_loader = DataLoader(dataset_val, 1, shuffle=False, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9,0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,50,300], gamma=0.5)

    accuracies_train = []
    losses_train = []
    accuracies_val = []
    losses_val = []

    epoch_start = 1
    training_epoch_start = 1
    if not args.finetuning_resume:
        checkpoint = torch.load(resume_path)
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            print("Could not load optimizer key")
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
        except:
            print("Could not load scheduler key")
        try:
            if not checkpoint.get('accuracies_val', None):
                epoch_start = int(checkpoint['epoch'])
            else:
                training_epoch_start = int(checkpoint['epoch'])
                accuracies_train = checkpoint['accuracies_train']
                losses_train = checkpoint['losses_train']
                accuracies_val = checkpoint['accuracies_val']
                losses_val = checkpoint['losses_val']
        except:
            print("Could not load epoch key")

    best_acc = 0
    save_every = 1
    if epochs > 10:
        save_every = 15

    with open(os.path.join(out_path,'parameters.txt'),'w') as f:
        f.write(str({k:v for k,v in optimizer.param_groups[0].items() if k != 'params'} ) + '\n')
        f.write(str(scheduler.state_dict()) + '\n')
        f.write("dataset:" + str(args.dataset) + '\n')
        f.write("holdout:" + str(args.valsplit) + '\n')
        f.write("training:" + str(args.trainsplits) + '\n')
        f.write("epoch:" + str(training_epoch_start) + '\n')


    for epoch in range(training_epoch_start, epochs + 1):
        start = time()
        acc_t,loss_t=method.train(epoch,model,train_loader, optimizer)
        acc_v,loss_v=method.validate(model,val_loader)
        
        accuracies_train.append(acc_t/100)
        losses_train.append(loss_t)
        accuracies_val.append(acc_v/100)
        losses_val.append(loss_v)

        best = False
        if acc_v > best_acc:
            best_acc = acc_v
            best = True
        
        if epoch % save_every == 0 or best:
            save_architecture(out_path, epoch, model, optimizer, scheduler, best, 
                losses_train = losses_train,
                accuracies_val = accuracies_val,
                losses_val = losses_val,
                accuracies_train = accuracies_train)
        make_plot(out_path, epoch, accuracies_train, losses_train, accuracies_val, losses_val, start_epoch = epoch_start)
        
        scheduler.step()

        end = time()

        taken = end-start
        print(f"====> Epoch took: {taken:.2f}s, lr: {scheduler.get_last_lr()}, ETA: {((taken) * (epochs - epoch)) / 60 / 60:.2f}h")

