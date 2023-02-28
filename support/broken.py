from captum.attr import DeepLift
import torch
import torchvision
from torchvision import transforms
import numpy as np
import cv2
import sys
import os
import copy
from tqdm import tqdm
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import dataman
import modman
import utils

class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        # std_inv = 1 / (std + 1e-7)
        std_inv = 1 / std
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
        #return super().__call__(tensor)

def image_normalize(image, dataset):
    if dataset == 'CIFAR10':
       transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    elif dataset == 'ImageNet':
        transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    elif dataset == 'MNIST':
        transform = transforms.Normalize((0.1307, ), (0.3081, ))
    else:
        raise NotImplementedError
    return transform(image)

def image_normalize_inv(image, dataset):
    if dataset == 'CIFAR10':
        transform = NormalizeInverse((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    elif dataset == 'ImageNet':
        transform = NormalizeInverse((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    elif dataset == 'MNIST':
        transform = NormalizeInverse((0.1307, ), (0.3081, ))
    else:
        raise NotImplementedError
    return transform(image)

def mask_ratio(src):
    src_mask = src == 255
    ones = np.ones(src.shape)
    return src_mask.astype(np.float32).sum() / ones.sum()

def mask_pos(attr):
    if isinstance(attr, torch.Tensor):
        pixel0 = attr.squeeze().cpu().detach().numpy()
    else:
        pixel0 = attr
    index = np.where(pixel0 > 0)[0]
    pixel0_pos = pixel0[index]
    pixel0_pos = pixel0_pos.mean(0)
    pixel0_pos -= pixel0_pos.min()
    pixel0_pos /= pixel0_pos.max()
    pixel0_pos *= 255
    pixel0_pos = pixel0_pos.astype(np.uint8)
    # _, th = cv2.threshold(pixel0_pos, int(255 * 0.6), 255, cv2.THRESH_BINARY)
    _, th = cv2.threshold(pixel0_pos, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    return th, closing

def segment_noise(org_tensor, mask):
    # print('org_tensor: ', org_tensor.size())
    if isinstance(org_tensor, torch.Tensor):
        org_np = org_tensor.squeeze().cpu().detach().numpy()
    else:
        org_np = org_tensor
    # print(mask.max())
    noise = np.random.normal(size=org_np.shape)
    for i in range(3):
        org_np[i] *= (mask != 255).astype(np.uint8)
        org_np[i] += (noise[i] * (mask == 255).astype(np.uint8))
    org_np = np.transpose(org_np, (1, 2, 0))
    # print('org_np: ', org_np.max())
    # org_np = cv2.cvtColor(org_np, cv2.COLOR_RGB2BGR)
    return org_np * 255

parser = argparse.ArgumentParser()
parser.add_argument("model_name", choices=['resnet50', 'vgg16_bn', 'densenet121', 'googlenet', 'inception_v3'])
parser.add_argument("output_dir", type=str)
parser.add_argument('--image-size', default=32)
parser.add_argument('--split', default='train')
args = parser.parse_args()

model_name = args.model_name
image_size = args.image_size
output_dir = args.output_dir
split = args.split
dataset = 'CIFAR10'
batch_size = 1

output_file = f'{output_dir}/dl-broken-{model_name}-{dataset}-{split}.pt'
utils.ensure_dir_of(output_file)

test_loader = dataman.get_benign_loader(dataset, image_size, split, batch_size)
torch_model = modman.get_torch_mod(model_name, dataset)

model_ex = copy.deepcopy(torch_model)
dl = DeepLift(model_ex)

ret = []

for i, (image, label) in enumerate(tqdm(test_loader)):
    image = image#.cuda()
    label = label#.cuda()

    logit = torch_model(image)
    pred = logit.argmax(-1)

    baselines = torch.zeros(image.size())#.cuda()
    attr = dl.attribute(image, target=pred, baselines=baselines)
    # attr = dl.attribute(image, target=label, baselines=baselines)

    with torch.no_grad():
        th, closing = mask_pos(attr[0])
        if mask_ratio(closing) >= 0.9:
            continue

        broken = segment_noise(image_normalize_inv(image, dataset), 255 - closing)
        broken = image_normalize(
            torch.from_numpy(broken / 255).transpose(0, 1).transpose(0, 2).unsqueeze(0),#.cuda(),
            dataset
        )

    ret.append(broken.numpy())

    if i % 500 == 0:
        torch.save(ret, output_file)
