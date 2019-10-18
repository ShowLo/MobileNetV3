# -*- coding: UTF-8 -*-

'''
Train the model
Ref: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
from mobileNetV3 import MobileNetV3
import argparse
import copy

from statistics import *
from EMA import EMA
from LabelSmoothing import LabelSmoothingLoss
from DataLoader import dataloaders

def train_model(args, model, dataloader, criterion, optimizer, scheduler, use_gpu):
    '''
    train the model
    '''

    # exponential moving average (using in val)
    ema = EMA(model, decay=args.ema_decay)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    device = torch.device('cuda' if use_gpu else 'cpu')

    for epoch in range(args.start_epoch, args.num_epochs):
        # statistical information
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(dataloader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step(epoch)
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()
                # apply EMA at validation stage
                ema.apply_shadow()

            running_loss = 0.0
            running_corrects = 0

            end = time.time()

            # Iterate over data
            for i, (inputs, labels) in enumerate(dataloader[phase]):
                # measure data loading time
                data_time.update(time.time() - end)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                    losses.update(loss.item(), inputs.size(0))
                    top1.update(acc1[0], inputs.size(0))
                    top5.update(acc5[0], inputs.size(0))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                batch_time.update(time.time() - end)
                time = time.time()

                if phase == 'train' and (i + 1) % args.print_freq == 0:
                    progress.display(i)
            
            if phase == 'train':
                # EMA update after training
                ema.update()
            else:
                # restore the origin parameters after val
                ema.restore()
            
            print(phase + ' * Loss {losses.avg:.2e} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(losses=losses, top1=top1, top5=top5))
                # statistics
                '''
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                batch_loss = running_loss / ((i + 1) * args.batch_size)
                batch_acc = running_corrects.double() / ((i + 1) * args.batch_size)

                if phase == 'train' and (i + 1) % args.print_freq == 0:
                    print('[Epoch {}/{}]-[batch:{}/{}] lr:{:.4f} {} Loss: {:.6f}  Acc: {:.4f}  Time: {:.4f} sec/batch'.format(
                          epoch + 1, num_epochs, i + 1, round(dataset_sizes[phase]/args.batch_size), scheduler.get_lr()[0], phase, batch_loss, batch_acc, (time.time()-tic_batch)/args.print_freq))
                    tic_batch = time.time()
                '''

            '''
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            with open(os.path.join(args.save_path, 'result.txt'), 'a') as f:
                f.write('Epoch:{}/{} {} Loss: {:.4f} Acc: {:.4f} \n'.format(epoch + 1, num_epochs, phase, epoch_loss, epoch_acc))
            '''

        if epoch % args.save_epoch_freq == 0:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model.state_dict(), os.path.join(args.save_path, "epoch_" + str(epoch) + ".pth"))
        
        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print('Best val Accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='PyTorch implementation of MobileNetV3')
    # Root catalog of images
    parser.add_argument('--data-dir', type=str, default='/media/data2/chenjiarong/ImageData')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-class', type=int, default=1000)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    #parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--print-freq', type=int, default=1000)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default='/media/data2/chenjiarong/MobileNetV3/model')
    parser.add_argument('--resume', type=str, default='', help='For training from one checkpoint')
    parser.add_argument('--start-epoch', type=int, default=0, help='Corresponding to the epoch of resume')
    parser.add_argument('--ema-decay', type=float, default=0.9999, help='The decay of exponential moving average ')
    parser.add_argument('--dataset', type=str, default='ImageNet', help='The dataset to be trained')
    args = parser.parse_args()

    # read data
    dataloaders = dataloaders(args)

    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    # get model
    model = MobileNetV3(mode='small', classes_num=args.num_class)

    if use_gpu:
        model = torch.nn.DataParallel(model)
        model.to(torch.device('cuda'))
    else:
        model.to(torch.device('cpu'))

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            model.load_state_dict(torch.load(args.resume))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    # define loss function
    # criterion = nn.CrossEntropyLoss()
    
    # using Label Smoothing
    criterion = LabelSmoothingLoss(args.classes_num, label_smoothing=0.1)

    # optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00004)
    optimizer_ft = optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)

    # Decay LR by a factor of 0.99 every 3 epoch
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.99)

    model = train_model(args=args,
                        model=model,
                        dataloader=dataloaders
                        criterion=criterion,
                        optimizer=optimizer_ft,
                        scheduler=exp_lr_scheduler,
                        use_gpu=use_gpu)
    torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model_wts.pth'))