import argparse
import os

PARSER = argparse.ArgumentParser(description='Run given network on the given split and store outpus + running times')

PARSER.add_argument('file', default='splits.json')
PARSER.add_argument('--resume','-r', default='reproductions/LEAStereo/run/Kitti15/best/best.pth')
PARSER.add_argument('--outdir','-o', default=os.path.join('..','predictions'))
PARSER.add_argument('--dataset','-d', default='kitti2015')
PARSER.add_argument('--datasetsplit','-ds', default='training')
PARSER.add_argument('--splitname', '-s', default='validation1')
PARSER.add_argument('--method', '-m', default='LEAStereo')

PARSER_TRAIN = argparse.ArgumentParser(description='Run given network on the given split and store outpus + running times')

PARSER_TRAIN.add_argument('file')
PARSER_TRAIN.add_argument('save', help="path to overwrite/save new weights to")
PARSER_TRAIN.add_argument('valsplit',help="hold out split name")
PARSER_TRAIN.add_argument('trainsplits',nargs="+", help="split names to be used for training")
PARSER_TRAIN.add_argument('--resume','-r', help="path to weights dir to resume from (None)")
PARSER_TRAIN.add_argument('--dataset','-d', default='kitti2015')
PARSER_TRAIN.add_argument('--method', '-m', default='LEAStereo')
PARSER_TRAIN.add_argument('--epochs', '-e', default=800, type=int)
PARSER_TRAIN.add_argument('--batch', '-b', default=1, type=int)
PARSER_TRAIN.add_argument('--learning_rate', '-lr', default=1e-3, type=float)
PARSER_TRAIN.add_argument('--finetuning_resume', '-fr', default=True, action="store_false" ,help="if true, optimizer and scheduler are not loaded from checkpoint (True)")