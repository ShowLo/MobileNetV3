# -*- coding: UTF-8 -*-

'''
Image dataset loader, including ImageNet, cifar10 and cifar100
'''

from torchvision import transforms, datasets
import os
import torch
from PIL import Image
import scipy.io as scio

def Cifar10DataLoader(args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    }

    image_datasets = {}
    image_datasets['train'] = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=data_transforms['train'])
    image_datasets['val'] = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=data_transforms['val'])

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True if x == 'train' else False,
                    num_workers=args.num_workers, pin_memory=True) for x in ['train', 'val']}
    
    return dataloders

def Cifar100DataLoader(args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])
    }

    image_datasets = {}
    image_datasets['train'] = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=data_transforms['train'])
    image_datasets['val'] = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=data_transforms['val'])

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True if x == 'train' else False,
                    num_workers=args.num_workers, pin_memory=True) for x in ['train', 'val']}
    
    return dataloders

def ImageNetDataLoader(args):
    # data transform
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(root=os.path.join(args.data_dir, 'ILSVRC2012_img_train'), transform=data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(root=os.path.join(args.data_dir, 'ILSVRC2012_img_val'), transform=data_transforms['val'])
    
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True if x == 'train' else False,
                    num_workers=args.num_workers, pin_memory=True) for x in ['train', 'val']}

    return dataloders

def dataloaders(args):
    dataset = args.dataset.lower()
    assert dataset in ['imagenet', 'cifar10', 'cifar100']
    if dataset == 'imagenet':
        return ImageNetDataLoader(args)
    elif dataset == 'cifar10':
        return Cifar10DataLoader(args)
    elif dataset == 'cifar100':
        return Cifar100DataLoader(args)