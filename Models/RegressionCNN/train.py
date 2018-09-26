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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode= 'min', verbose= True, patience= 8)
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
    accuracy1 = 0
    accuracy4 = 0
    correct1 = 0
    correct4 = 0

    correct4_list = []
    correct1_list = []

    target4_list = []
    target1_list = []

    # validate the model over the test set and record no gradient history
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(test_loader):

            input, target = input.to(device), target.type(torch.FloatTensor).to(device)

            output = model(input)

            # print the predictions and expected values
            # for pred, ex in zip(output, target):
            #     print('pred: {:.8f}, {:.8f} expec: {:.8f}, {:.8f} \n'.format(pred[0].item(), pred[1].item(), ex[0].item(), ex[1].item()))

            # calculate the correctness of the prediction within a 1 and 4 pixel range
            pred = target - output

            # enumerate over the predictions and check if they are within a range
            # add the correct outputs of the net to a list and the targets
            for idx, p in enumerate(pred):
                if (p >= -.0625).all() and (p <= .0625).all():
                    correct4 += 1
                    correct4_list.append(output[idx])
                    target4_list.append(target[idx])

                if (p >= -.015625).all() and (p <= .015625).all():
                    correct1 += 1
                    correct1_list.append(output[idx])
                    target1_list.append(target[idx])

            # dirty way of iterating over the lists and printing their contents
            print("Predictions correct within 4-pixels\n")
            for corr4, tar4 in zip(correct4_list, target4_list):
                pred4 = tar4 - corr4
                print("prediction : {:.8f}, {:.8f} target: {:.8f}, {:.8f}".format(corr4[0].item(), corr4[1].item(), tar4[0].item(), tar4[1].item()))
                print("difference between the two points: {:.8f}, {:.8f}\n".format(pred4[0].item(), pred4[1].item()))


            print("Predictions correct within 1-pixel\n")
            for corr1, tar1 in zip(correct1_list, target1_list):
                pred1 = tar1 - corr1
                print("prediction : {:.8f}, {:.8f} target: {:.8f}, {:.8f}".format(corr1[0].item(), corr1[1].item(), tar1[0].item(), tar1[1].item()))
                print("difference between the two points: {:.8f}, {:.8f}\n".format(pred1[0].item(), pred1[1].item()))

            print("Correct number of predictions in this batch with a 4-pixel range: {}/{} \n".format(len(correct4_list), args.batch_size))
            print("Correct number of predictions in this batch with a 1-pixel range: {}/{} \n".format(len(correct1_list), args.batch_size))

            # Calculate the RMSE loss
            if args.loss_fn == 'MSE':
                test_loss += F.mse_loss(output, target).item()
            elif args.loss_fn == 'RMSE':
                test_loss += RMSELoss.rmse_loss(output, target).item()

            # clear the lists for the next iteration
            correct4_list.clear()
            correct1_list.clear()
            target1_list.clear()
            target4_list.clear()

    test_loss /= (len(test_loader.dataset))
    accuracy1 = 100. * correct1 / len(test_loader.dataset)
    accuracy4 = 100. * correct4 / len(test_loader.dataset)
    print('{} Test Loss: {:.7f}, Accuracy 4-pixels: {}/{} ({:.3f}%), Accuracy 1-pixel: {}/{} ({:.3f}%) \n'
        .format(args.loss_fn, test_loss, correct4, len(test_loader.dataset), accuracy4,
            correct1, len(test_loader.dataset), accuracy1))

    return test_loss, accuracy1

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