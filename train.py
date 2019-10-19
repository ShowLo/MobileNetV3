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
from ResultWriter import ResultWriter

def train_model(args, model, dataloader, criterion, optimizer, scheduler, use_gpu):
    '''
    train the model
    '''

    # exponential moving average (using in val)
    ema = EMA(model, decay=args.ema_decay)
    ema.register()

    # save result every epoch
    train_file_name = 'train.csv'
    val_file_name = 'val.csv'
    resultTrainWriter = ResultWriter(args.save_path, train_file_name)
    resultValWriter = ResultWriter(args.save_path, val_file_name)
    if args.start_epoch == 0:
        resultTrainWriter.create_csv(['epoch', 'loss', 'top-1', 'top-5'])
        resultValWriter.create_csv(['epoch', 'loss', 'top-1', 'top-5'])

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
            len(dataloader['train']),
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
                    # _, preds = torch.max(outputs, 1)
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
                end = time.time()

                if phase == 'train' and i % args.print_freq == 0:
                    progress.display(i)
            
            if phase == 'train':
                # EMA update after training
                ema.update()
                # write training result to file
                resultTrainWriter.write_csv([epoch, losses.avg, top1.avg.item(), top5.avg.item()])
            else:
                # restore the origin parameters after val
                ema.restore()
                # write val result to file
                resultValWriter.write_csv([epoch, losses.avg, top1.avg.item(), top5.avg.item()])
            
            print(phase.center(5) + ' ***    Loss:{losses.avg:.2e}    Acc@1:{top1.avg:.2f}    Acc@5:{top5.avg:.2f}'.format(losses=losses, top1=top1, top5=top5))

        if epoch % args.save_epoch_freq == 0:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model.state_dict(), os.path.join(args.save_path, "epoch_" + str(epoch) + ".pth"))
        
        # deep copy the model
        if phase == 'val' and top1.avg.item() > best_acc:
            best_acc = top1.avg.item()
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
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    #parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--print-freq', type=int, default=1000)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default='/media/data2/chenjiarong/saved-model')
    parser.add_argument('--resume', type=str, default='', help='For training from one checkpoint')
    parser.add_argument('--start-epoch', type=int, default=0, help='Corresponding to the epoch of resume')
    parser.add_argument('--ema-decay', type=float, default=0.9999, help='The decay of exponential moving average ')
    parser.add_argument('--dataset', type=str, default='ImageNet', help='The dataset to be trained')
    parser.add_argument('--mode', type=str, default='large', help='large or small MobileNetV3')
    # parser.add_argument('--num-class', type=int, default=1000)
    parser.add_argument('--width-multiplier', type=float, default=1.0, help='width multiplier')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    args = parser.parse_args()

    # folder to save what we need in this type: MobileNetV3-mode-dataset-width_multiplier-dropout-lr-batch_size-ema_decay
    folder_name = ['MobileNetV3', args.mode, args.dataset, str(args.width_multiplier), str(args.dropout), str(args.lr), str(args.batch_size), str(args.ema_decay)]
    folder_name = '-'.join(folder_name)
    args.save_path = os.path.join(args.save_path, folder_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # read data
    dataloaders = dataloaders(args)

    # different input size and number of classes for different datasets
    # (default: ImageNet)
    input_size = 224
    num_class = 1000
    if args.dataset.lower() == 'cifar10':
        input_size = 32
        num_class = 10
    elif args.dataset.lower() == 'cifar100':
        input_size = 32
        num_class = 100
        
    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    # get model
    model = MobileNetV3(mode=args.mode, classes_num=num_class, input_size=input_size, width_multiplier=args.width_multiplier, dropout=args.dropout)

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
    criterion = LabelSmoothingLoss(num_class, label_smoothing=0.1)

    # optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00004)
    optimizer_ft = optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)

    # Decay LR by a factor of 0.99 every 3 epoch
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.99)

    model = train_model(args=args,
                        model=model,
                        dataloader=dataloaders,
                        criterion=criterion,
                        optimizer=optimizer_ft,
                        scheduler=exp_lr_scheduler,
                        use_gpu=use_gpu)
    torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model_wts.pth'))