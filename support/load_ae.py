import os
import copy
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset

import torchvision


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['ImageNet', 'CIFAR10'])
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--metric', type=str, default='NC',
                    choices=['NC', 'NCS', 'KMN', 'SNA', 'NBC', 'TopK', 'TopKPatt', 'TFC'])
parser.add_argument('--alg', type=str, default='PGD',
                    choices=['Feat', 'PGD', 'CW', 'FGSM'])
parser.add_argument('--output_root', type=str, default='/data/alex/output/Coverage/')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--image_size', type=int, default=128)

parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--NC_threshold', type=float, default=0.75)
parser.add_argument('--TFC_threshold', type=float, default=50)
parser.add_argument('--KMN_K', type=int, default=60)
parser.add_argument('--TopK_K', type=int, default=2)
parser.add_argument('--num_cat', type=float, default=1)
parser.add_argument('--num_perclass', type=float, default=1)
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


args.model = 'resnet50'
args.dataset = 'CIFAR10'

split = 'test' # ae generated using test data, or set split = 'train'
if args.alg in ['PGD', 'FGSM', 'CW']:
    name = '%s-%s-%s-%s.pt' % (args.alg, args.model, args.dataset, split)
    adv_images, labels = torch.load('/export/d2/alex/dataset/adversarial_examples/' + name)
    adv_data = TensorDataset(adv_images, labels)
    ae_loader = torch.utils.data.DataLoader(
            adv_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False
        )


for i, (image, label) in enumerate(tqdm(ae_loader)):
    image = image.cuda()
    label = label.cuda()
    print('image size: ', image.size())
    break