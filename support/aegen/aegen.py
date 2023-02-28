#! /usr/bin/env python3

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

import argparse
import torchattacks

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import modman
import dataman

class TorchCIFAR10Dataset(Dataset):
    def __init__(self,
                 args,
                 image_dir='/dev/shm/deployed-datasets/cifar-10-png/',
                 split='train'):
        super(TorchCIFAR10Dataset).__init__()
        self.args = args
        # self.image_dir = image_dir + ('train/' if split == 'train' else 'test/')
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(self.args.image_size),
                        transforms.CenterCrop(self.args.image_size),
                        transforms.ToTensor(),
                        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
                ])
        # self.norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        self.image_list = []
        self.cat_list = sorted(os.listdir(self.image_dir))[:self.args.num_cat]
        for cat in self.cat_list:
            name_list = sorted(os.listdir(self.image_dir + cat))
            self.image_list += [self.image_dir + cat + '/' + image_name for image_name in name_list]
        print('Total %d Data.' % len(self.image_list))
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2] # label name
        label = self.cat_list.index(label)
        label = torch.LongTensor([label]).squeeze()
        image = Image.open(image_path)#.convert('RGB')
        image = self.transform(image)
        return image, label

class DataLoader(object):
    def __init__(self, args):
        self.args = args
        self.init_param()
        #self.init_dataset()

    def init_param(self):
        self.gpus = torch.cuda.device_count()
        # self.transform = transforms.Compose([
        #                         transforms.Resize(self.args.image_size),
        #                         transforms.ToTensor(),
        #                         transforms.Normalize((0.5,), (0.5,)),
        #                    ])

    def get_loader(self, dataset, shuffle=True):
        data_loader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=self.args.batch_size * self.gpus,
                            num_workers=int(self.args.num_workers),
                            shuffle=shuffle
                        )
        return data_loader

def ensure_dir_of(filepath):
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                            choices=['CIFAR10', 'ImageNet'])
    parser.add_argument('--model', type=str, default='resnet50',
                            choices=['resnet50', 'vgg16_bn', 'mobilenet_v2', 'densenet121', 'googlenet', 'inception_v3'])
    parser.add_argument('--alg', type=str, default='PGD',
                            choices=['FGSM', 'BIM', 'CW', 'DeepFool', 'PGD'])
    parser.add_argument('--is_train', type=int, default=1,
                            choices=[0, 1])
    args = parser.parse_args()
    args.batch_size = 64
    args.num_workers = 4

    model = modman.get_torch_mod(args.model, args.dataset).cuda()

    if args.dataset == 'ImageNet':
        raise NotImplementedError
        args.z_dim = 120
        args.image_size = 128
        args.num_cat = 100
        train_data = TorchImageNetDatasetCov(args, 'train')
        test_data = TorchImageNetDatasetCov(args, 'val')
        # assert args.num_cat <= 1000
    elif args.dataset == 'CIFAR10':
        args.z_dim = 128
        args.image_size = 32
        args.num_cat = 10
        train_data = TorchCIFAR10Dataset(args, split='train')
        test_data = TorchCIFAR10Dataset(args, split='test')
    loader = DataLoader(args)
    train_loader = loader.get_loader(train_data, False)
    test_loader = loader.get_loader(test_data, False)

    atk = getattr(torchattacks, args.alg)(model)
    name = f'{args.alg}-{args.model}-{args.dataset}-{"train" if args.is_train else "test"}.pt'
    save_path = "./adversarial_examples/" + name
    ensure_dir_of(save_path)
    if args.is_train:
        atk.save(train_loader, save_path=save_path, verbose=True)
    else:
        atk.save(test_loader, save_path=save_path, verbose=True)
