import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
import torch.backends.cudnn as cudnn
import time
import sys
import shutil
import CannulaDataset
import RMSELoss
import numpy as np

def train(args, model, device, checkpoint):

    # Data transformations
    if args.resize is not None:
        data_transform = transforms.Compose([
            transforms.Resize((args.resize, args.resize)),
            transforms.ToTensor()            
            ])
    else:
        # data_transform = transforms.Compose([
        #     transforms.Normalize([0.016813556], [.012097757])
        #     ])
        data_transform = None

    if args.resize is not None:
        print("\nImages resized to %d x %d" % (args.resize, args.resize))

    # load the data for k-folds
    inputs = np.load(os.path.join(args.root_dir, args.inputs))
    inputs = np.expand_dims(inputs, 3)
    targets = np.load(os.path.join(args.root_dir, args.targets))
    kfold = kFold(inputs, targets)

    train_dataset = CannulaDataset.CannulaDataset(
        inputs, 
        targets, 
        kfold[0][1][0], 
        transform= data_transform
        )

    test_dataset = CannulaDataset.CannulaDataset(
        inputs, 
        targets, 
        kfold[0][1][1], 
        transform= data_transform
        )

    # use the torch dataloader class to enumerate over the data during training
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size= args.batch_size, 
        shuffle= True, 
        num_workers= args.num_processes,
        pin_memory= True
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size= args.batch_size,
        shuffle= True,
        num_workers = args.num_processes,
        pin_memory= True
        )

    # set the optimizer depending on choice
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr= args.lr, momentum= args.momentum, dampening=0, weight_decay= 0 if args.weight_decay is None else args.weight_decay, nesterov= False)
    elif args.optimizer == 'AdaG':
        optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
    elif args.optimizer == 'AdaD':
        optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    elif args.optimizer == 'RMS':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay= 0 if args.weight_decay is None else args.weight_decay, momentum=args.momentum, centered=False)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    print("\nUsing optimizer: %s" % (args.optimizer))

    if args.loss_fn == 'MSE':
        criterion = torch.nn.MSELoss().cuda() if device == "cuda" else torch.nn.MSELoss()
    elif args.loss_fn == 'RMSE':
        criterion = RMSELoss.RMSELoss().cuda() if device == "cuda" else RMSELoss.RMSELoss()

    # either take the minimum loss then reduce LR or take max of accuracy then reduce LR
    if args.plateau == 'loss':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode= 'min', verbose= True, patience= 10)
    elif args.plateau == 'accuracy':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode= 'max', verbose= True, patience= 6)

    print("\nReducing learning rate on %s plateau\n" % (args.plateau))

    best_prec1 = 0 if checkpoint is None else checkpoint['best_prec1']
    is_best = False

    del checkpoint

    # train and validate the model accordingly
    total_time = time.clock()
    for epoch in range(args.start_epoch, args.epochs + 1):
        train_epoch(epoch, args, model, optimizer, criterion, train_loader, device)
        test_loss, accuracy = test_epoch(model, test_loader, device, args)

        if args.plateau == 'loss':
            scheduler.step(test_loss)
        elif args.plateau == 'accuracy':
            scheduler.step(accuracy)

        if accuracy > best_prec1:
            best_prec1 = accuracy
            is_best = True

        # save the model every epoch
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            'time': time.clock() - total_time
        }, is_best)

        is_best = False

def train_epoch(epoch, args, model, optimizer, criterion, train_loader, device):
    model.train()
    correct = 0

    # train the model over the training set
    for batch_idx, (input, target) in enumerate(train_loader):
        
        input, target = input.to(device), target.type(torch.FloatTensor).to(device)

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # report the train metrics depending on the log interval
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(input), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    print('')

def test_epoch(model, test_loader, device, args):
    model.eval()
    test_loss = 0
    accuracy = 0
    correct = 0

    # validate the model over the test set and record no gradient history
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(test_loader):

            input, target = input.to(device), target.type(torch.FloatTensor).to(device)

            output = model(input)

            for pred, ex in zip(output, target):
                print('pred: {:.8f}, {:.8f} expec: {:.8f}, {:.8f} \n'.format(pred[0].item(), pred[1].item(), ex[0].item(), ex[1].item()))
            # Calculate the RMSE loss
            if args.loss_fn == 'MSE':
                test_loss += F.mse_loss(output, target).item()
            elif args.loss_fn == 'RMSE':
                test_loss += RMSELoss.rmse_loss(output, target).item()

    test_loss /= (len(test_loader.dataset))
    print('{} Test Loss: {:.7f}'.format(args.loss_fn, test_loss))
    # accuracy = 100. * correct / len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'
    #       .format(test_loss, correct, len(test_loader.dataset),
    #               100. * correct / len(test_loader.dataset)))

    return test_loss, accuracy

# Saves the model as a checkpoint
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def kFold(inputs, targets):
        kfold = KFold(5, True, 11)
        idxs = []

        for train, test in enumerate(kfold.split(inputs, targets)):
            idxs.append((train, test))

        return idxs