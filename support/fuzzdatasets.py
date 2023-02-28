# Dataset module for fuzzer

import os
import sys
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

# from gan_cifar10 import load_Gen as get_cifar10_G

from . import my_utils

class CIFAR10Dataset(object):
    def __init__(self,
                 args,
                 image_dir='/dev/shm/deployed-datasets/cifar-10-png/',
                 split='train'):
        self.args = args
        # self.image_dir = image_dir + ('train/' if split == 'train' else 'test/')
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(self.args.image_size),
                        transforms.CenterCrop(self.args.image_size),
                        transforms.ToTensor(),
                        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
                ])
        self.norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        self.image_list = []
        self.cat_list = sorted(os.listdir(self.image_dir))[:self.args.num_cat]
        for cat in self.cat_list:
            name_list = sorted(os.listdir(self.image_dir + cat))[:self.args.num_perclass]
            self.image_list += [self.image_dir + cat + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def get_len(self):
        return len(self.image_list)

    def get_item(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2] # label name
        label = self.cat_list.index(label)
        label = torch.LongTensor([label]).squeeze()

        image = Image.open(image_path)#.convert('RGB')
        image = self.transform(image)
        return (image, label)

    def build(self):
        image_list = []
        label_list = []
        for i in tqdm(range(self.get_len())):
            (image, label) = self.get_item(i)
            image_list.append(image)
            label_list.append(label)
            # if len(image_list) % 3000 == 0:
            #     print('Building %d data' % len(image_list))
        return image_list, label_list

    def to_numpy(self, image_list, is_image=True):
        image_numpy_list = []
        for i in tqdm(range(len(image_list))):
            # if i % self.args.num_perclass < self.args.num_perclass * 0.8: # FIXME
            image = image_list[i]
            if is_image:
                image_numpy = image.transpose(0, 2).numpy()
            else:
                image_numpy = image.numpy()
            image_numpy_list.append(image_numpy)
        print('Numpy: %d' % len(image_numpy_list))
        return image_numpy_list

    def to_batch(self, data_list, is_image=True):
        batch_list = []
        batch = []
        for i, data in enumerate(data_list):
            if i and i % self.args.batch_size == 0:
                batch_list.append(torch.stack(batch, 0))
                batch = []

            batch.append(self.norm(data) if is_image else data)
        if len(batch):
            batch_list.append(torch.stack(batch, 0))
        print('Batch: %d' % len(batch_list))
        return batch_list



class ImageNetDataset(object):
    def __init__(self,
                 args,
                 image_dir='/dev/shm/deployed-datasets/CLS-LOC/',
                 label_file='SelectedLabel-100K.json',
                 label2index_file='/export/shm/deployed-datasets/CLS-LOC/ImageNetLabel2Index.json',
                 split='train'):
        self.args = args
        # self.image_dir = image_dir + ('train/' if split == 'train' else 'test/')
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(self.args.image_size),
                        transforms.CenterCrop(self.args.image_size),
                        transforms.ToTensor(),
                        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        self.norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.image_list = []

        with open(label2index_file, 'r') as f:
            self.label2index = json.load(f)

        with open(label_file, 'r') as f:
            self.cat_list = json.load(f)

        for cat in self.cat_list:
            name_list = sorted(os.listdir(self.image_dir + cat))[:self.args.num_perclass]
            self.image_list += [self.image_dir + cat + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def get_len(self):
        return len(self.image_list)

    def get_item(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2] # label name
        # label = self.cat_list.index(label)
        index = self.label2index[label]
        index = torch.LongTensor([index]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return (image, index)

    def build(self):
        image_list = []
        label_list = []
        for i in tqdm(range(self.get_len())):
            (image, label) = self.get_item(i)
            image_list.append(image)
            label_list.append(label)
            # if len(image_list) % 3000 == 0:
            #     print('Building %d data' % len(image_list))
        return image_list, label_list

    def to_numpy(self, image_list, is_image=True):
        image_numpy_list = []
        for i in tqdm(range(len(image_list))):
            # if i % self.args.num_perclass < self.args.num_perclass * 0.8: # FIXME
            image = image_list[i]
            if is_image:
                image_numpy = image.transpose(0, 2).numpy()
            else:
                image_numpy = image.numpy()
            image_numpy_list.append(image_numpy)
        print('Numpy: %d' % len(image_numpy_list))
        return image_numpy_list

    def to_batch(self, data_list, is_image=True):
        batch_list = []
        batch = []
        for i, data in enumerate(data_list):
            if i and i % self.args.batch_size == 0:
                batch_list.append(torch.stack(batch, 0))
                batch = []
            batch.append(self.norm(data) if is_image else data)

        if len(batch):
            batch_list.append(torch.stack(batch, 0))
        print('Batch: %d' % len(batch_list))
        return batch_list

class ImageNetDatasetCov(object):
    def __init__(self,
                 args,
                 image_dir='/dev/shm/deployed-datasets/CLS-LOC/',
                 # label_file='SelectedLabel-100K.json',
                 label2index_file='/export/shm/deployed-datasets/CLS-LOC/ImageNetLabel2Index.json',
                 split='train'):
        self.args = args
        # self.image_dir = image_dir + ('train/' if split == 'train' else 'test/')
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(self.args.image_size),
                        transforms.CenterCrop(self.args.image_size),
                        transforms.ToTensor(),
                        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        self.norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.image_list = []

        with open(label2index_file, 'r') as f:
            self.label2index = json.load(f)

        self.cat_list = sorted(os.listdir(self.image_dir))[:args.num_cat]

        for cat in self.cat_list:
            name_list = sorted(os.listdir(self.image_dir + cat))[:self.args.num_perclass]
            self.image_list += [self.image_dir + cat + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def get_len(self):
        return len(self.image_list)

    def get_item(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2] # label name
        # label = self.cat_list.index(label)
        index = self.label2index[label]
        assert int(index) < self.args.num_cat
        index = torch.LongTensor([index]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return (image, index)

    def build(self):
        image_list = []
        label_list = []
        for i in tqdm(range(self.get_len())):
            (image, label) = self.get_item(i)
            image_list.append(image)
            label_list.append(label)
            # if len(image_list) % 3000 == 0:
            #     print('Building %d data' % len(image_list))
        return image_list, label_list

    def to_numpy(self, image_list, is_image=True):
        image_numpy_list = []
        for i in tqdm(range(len(image_list))):
            # if i % self.args.num_perclass < self.args.num_perclass * 0.8: # FIXME
            image = image_list[i]
            if is_image:
                image_numpy = image.transpose(0, 2).numpy()
            else:
                image_numpy = image.numpy()
            image_numpy_list.append(image_numpy)
        print('Numpy: %d' % len(image_numpy_list))
        return image_numpy_list

    def to_batch(self, data_list, is_image=True):
        batch_list = []
        batch = []
        for i, data in enumerate(data_list):
            if i and i % self.args.batch_size == 0:
                batch_list.append(torch.stack(batch, 0))
                batch = []
            batch.append(self.norm(data) if is_image else data)

        if len(batch):
            batch_list.append(torch.stack(batch, 0))
        print('Batch: %d' % len(batch_list))
        return batch_list

class TorchImageNetDatasetCov(Dataset):
    def __init__(self,
                 args,
                 image_dir='/dev/shm/deployed-datasets/CLS-LOC/',
                 # label_file='SelectedLabel-100K.json',
                 label2index_file='/export/shm/deployed-datasets/CLS-LOC/ImageNetLabel2Index.json',
                 split='train'):
        super(TorchImageNetDatasetCov).__init__()
        self.args = args
        # self.image_dir = image_dir + ('train/' if split == 'train' else 'test/')
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(self.args.image_size),
                        transforms.CenterCrop(self.args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        # self.norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.image_list = []

        with open(label2index_file, 'r') as f:
            self.label2index = json.load(f)

        self.cat_list = sorted(os.listdir(self.image_dir))[:args.num_cat]

        for cat in self.cat_list:
            name_list = sorted(os.listdir(self.image_dir + cat))
            self.image_list += [self.image_dir + cat + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2] # label name
        # label = self.cat_list.index(label)
        index = self.label2index[label]
        assert int(index) < self.args.num_cat
        index = torch.LongTensor([index]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index

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
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
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


if __name__ == '__main__':
    pass
