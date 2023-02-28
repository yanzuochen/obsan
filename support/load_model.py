import os
import copy
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from data_loader import ImageNetDataset, CIFAR10Dataset, DataLoader
import neuron_coverage as tool
import classifier as cifar10_models
from criteria import *

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['ImageNet', 'CIFAR10'])
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--metric', type=str, default='NBC',
                    choices=['Stat', 'NC', 'NCS', 'KMN', 'SNA', 'NBC', 'TopK', 'TopKPatt', 'TFC'])
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--image_size', type=int, default=128)

parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--NC_threshold', type=float, default=0.75)
parser.add_argument('--TFC_threshold', type=float, default=50)
parser.add_argument('--KMN_K', type=int, default=60)
parser.add_argument('--TopK_K', type=int, default=2)
parser.add_argument('--z_dim', type=int, default=120)
parser.add_argument('--save_every', type=int, default=50)
parser.add_argument('--use_stat', type=int, default=0)
parser.add_argument('--use_trunc', type=int, default=0)
parser.add_argument('--truncation', type=float, default=-1)
parser.add_argument('--hyper', type=float, default=-1)

args = parser.parse_args()

# manual_seed = 1234
# random.seed(manual_seed)
# torch.manual_seed(manual_seed)
# torch.cuda.manual_seed_all(manual_seed)

if args.dataset == 'ImageNet':
    model = torchvision.models.__dict__[args.model](pretrained=False)
    path = ('/export/d2/alex/dataset/TORCH_HOME/%s.pth' % args.model)
    model.eval()
    model.load_state_dict(torch.load(path))
    args.image_size = 128
elif args.dataset == 'CIFAR10':
    model = getattr(cifar10_models, args.model)(pretrained=False)
    path = ('/home/alex/CIFAR-classifier/cifar10/%s/%s.pt' % (args.model, args.model))
    model.eval()
    model.load_state_dict(torch.load(path))
    args.image_size = 32

model = torch.nn.DataParallel(model).cuda()
model.eval()

input_size = (1, args.nc, args.image_size, args.image_size)
random_data = torch.randn(input_size).cuda()
layer_size_dict = tool.get_layer_output_sizes(model, random_data)


# if args.dataset == 'ImageNet':
#     train_data = ImageNetDataset(args, split='train')
#     test_data = ImageNetDataset(args, split='val')
# elif args.dataset == 'CIFAR10':
#     train_data = CIFAR10Dataset(args, split='train')
#     test_data = CIFAR10Dataset(args, split='test')

# loader = DataLoader(args)
# train_loader = loader.get_loader(train_data, False)
# test_loader = loader.get_loader(test_data, False)


CoveDict = {
    'NC': LayerNC,
    'NCS': LayerNCS,
    'KMN': LayerKMN,
    'SNA': LayerSNA,
    'NBC': LayerNBC,
    'TopK': LayerTopK,
    'TopKPatt': LayerTopKPattern,
    'TFC': LayerTFC
}

ParamDict = {
    'NC': args.hyper,
    'NCS': args.hyper,
    'KMN': int(args.hyper),
    'SNA': int(args.hyper),
    'NBC': int(args.hyper),
    'TopK': int(args.hyper),
    'TopKPatt': int(args.hyper),
    'TFC': args.hyper
}

coverage = CoveDict[args.metric](model, ParamDict[args.metric], layer_size_dict)
print('Initial: %d' % coverage.current)

# if args.metric in ['KMN', 'NBC', 'SNA']:
#     print('Set range')
#     for i, (image, label) in enumerate(tqdm(train_loader)):
#         coverage.set_range(image.cuda())

# for i, (image, label) in enumerate(tqdm(train_loader)):
#     if args.use_stat:
#         label = torch.zeros(label.size()).type(torch.cuda.LongTensor)
#         coverage.build_step(image.cuda(), label.cuda())
#     else:
#         coverage.build_step(image.cuda())
#     # print(coverage.current)
# print('Build: %f' % coverage.current)

cov_path = '/export/d1/alex/aesan-data/Coverage/%s-%s-%s.pth' % (args.dataset, args.model, args.metric)
coverage.load(cov_path)
print('coverage: ', coverage.current)
if args.metric in ['KMN', 'NBC', 'SNA']:
    print('range_dict: ', coverage.range_dict.keys())
    # {'layer_name': [min, max]}