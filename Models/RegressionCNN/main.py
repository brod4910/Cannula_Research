# ugly importing of make_model. Not recommended for any other circumstance
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import make_model
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import model
import models
from train import train

def CreateArgsParser():
    parser = argparse.ArgumentParser(description='Lensless Camera Pytorch')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument('--resize', type=int, default=None, 
                    help='dimensions of both height and width to be resized')
    parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
    parser.add_argument('--weight-decay', type=int, default=None, metavar='N',
                    help='L2 decay (default: None)')
    # parser.add_argument('--lr-scheduler', action='store_true')
    parser.add_argument('--plateau', default= 'loss', 
                    help= 'Measurement to plateau on. Either loss or accuracy')
    parser.add_argument('--architecture', default= 'deep', 
                    help= 'Model architecture to use. Options: deep and wide. (default: deep)')
    parser.add_argument('--optimizer', default= 'SGD', 
                    help= 'Type of optimizer to use. Options: SGD, AdaG, AdaD, Adam, RMS')
    parser.add_argument('--root-dir', required= True,  
                    help='root directory where enclosing image files are located')
    parser.add_argument('--inputs', required= True, 
                    help='path to the location of the test csv')
    parser.add_argument('--targets', required= True, 
                    help='path to the location of the test csv')
    parser.add_argument('--resume', default= None, 
                    help='file to load checkpoint from')
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--loss-fn', default='RMSE',
                    help='Loss funciton to be used: RMSE, MSE')
    parser.add_argument('--f-layers', required= True, default=None,
                    help='Feature layers to be used during training. List of feature layers are in models.py.')
    parser.add_argument('--c-layers', required= True, default=None,
                    help='Classifying layers to be used during training. List of classifier layers are in models.py.')
    return parser

def main():
    args = CreateArgsParser().parse_args()
    checkpoint = None

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda is True:
        cudnn.benchmark = True

    # Create the network
    network = model.Model(make_model.make_layers(models.feature_layers[args.f_layers]), 
        make_model.make_classifier_layers(models.classifier_layers[args.c_layers]))

    print("Using feature layers:", args.f_layers)
    print("Using classifying layers:", args.c_layers)

    print("\nLearning rate: ", args.lr)

    # reload the checkpoint if needed
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            network.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if torch.cuda.device_count() > 1:
        print("===> Number of GPU's available: %d" % torch.cuda.device_count())
        network = nn.DataParallel(network)

    network = network.to(device)

    print("\nBatch Size: %d" % (args.batch_size))

    train(args, network, device, checkpoint)


if __name__ == '__main__':
    main()