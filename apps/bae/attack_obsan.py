#! /usr/bin/env python3

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import random
import numpy as np
import torch
import argparse
from torchvision import transforms
from tqdm import tqdm

from autoattack import AutoAttack

import inst
import modman
import evalutils
import dataman
import evalconfig as ec

dataset = 'CIFAR10'
seed_split = 'test'

model_name = 'Qresnet50'
prune_frac = 0.2
data_frac = .01
data_start_frac = 0.

batch_size = 1
image_size = 32

norm = 'Linf'
eps = .3
nqueries = 50

thresholds = {
    # 'NBC': None,
    # 'gn2': None,
    'NBC': (0, 0.007482122915098443),  # RN default
    'gn2': [14.5505566, 36.88809118],  # RN default
}

modes = ['none'] + list(thresholds.keys()) + ['+'.join(thresholds.keys())]
assert len(modes) == 4

scores = []

# https://stackoverflow.com/questions/107705/disable-output-buffering
class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

def populate_threshold_dict(mode, output_defs):
    if thresholds[mode] is not None:
        return

    sam_train_loader = dataman.get_sampling_benign_loader(dataset, image_size, 'train', batch_size, ec.sam_train_frac)
    rtmod = modman.load_module(f'../../built/{model_name}-{dataset}-{mode}-False.so')
    thresholds[mode] = evalutils.estimate_threshold(
        model_name, mode, rtmod, output_defs[:2], sam_train_loader
    )

def get_forward_fn(wrmod, mode):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    safe = lambda cov, t: t[0] <= cov <= t[1]
    if mode != 'none':
        tlist = list(thresholds.values()) if mode == modes[-1] else [thresholds[mode]]
    def forward(x):
        x = normalize(x)
        logits, *covs = wrmod.run(x, rettype='all')
        if covs:
            scores.append(np.array(covs).reshape(-1,))
        if mode == 'none' or all(safe(c, t) for c, t in zip(covs, tlist)):
            return torch.tensor(logits)
        return None
    return forward

def launch_square_attack(forward_fn, images, labels):
    scores.clear()
    adversary = AutoAttack(
        forward_fn, norm=norm, eps=eps, version='standard', device='cpu'
    )
    adversary.attacks_to_run = ['square']
    adversary.square.n_queries = nqueries
    adversary.square.verbose = True
    dict_adv = adversary.run_standard_evaluation(images, labels, bs=batch_size)
    torch.save(np.array(scores), f'../../results/bae/scores-{mode}-{eps}-{nqueries}.pt')
    torch.save(dict_adv, f'../../results/bae/images-{mode}-{eps}.pt')
    return dict_adv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=modes)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--eps', type=float, default=eps)
    parser.add_argument('--nqueries', type=int, default=nqueries)
    args = parser.parse_args()

    random_seed = args.random_seed
    mode = args.mode
    eps, nqueries = args.eps, args.nqueries
    hybrid = mode == modes[-1]
    protected = mode != 'none'
    is_gn = mode in inst.gn_modes

    print(f'Running with {mode=}, {random_seed=}, {eps=}, {nqueries=}')

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    data_loader = dataman.get_sampling_benign_loader(
        f'{dataset}RAW', image_size, seed_split, batch_size, data_frac,
        start_frac=data_start_frac
    )

    mod_path = '../../' if hybrid or is_gn or not protected else '../../results/sel/'
    suffix = f'-{prune_frac}' if protected and not is_gn else '-False'
    mod_path += f'built/{model_name}-{dataset}-{mode}{suffix}.so'
    rtmod = modman.load_module(mod_path)

    output_defs = [{'shape': [1, 10], 'dtype': 'float32'}] + \
        [{'shape': [1], 'dtype': 'float32'}] * (int(protected) + int(hybrid))
    wrmod = modman.WrappedRtMod(rtmod, output_defs)

    xsys = [d for d in tqdm(data_loader)]
    images, labels = torch.stack([e[0].squeeze(0) for e in xsys]), torch.concat([e[1] for e in xsys])

    if protected:
        if hybrid:
            [populate_threshold_dict(m, output_defs) for m in thresholds]
        else:
            populate_threshold_dict(mode, output_defs)

    forward_fn = get_forward_fn(wrmod, mode)
    print(f'#FP: {sum(forward_fn(x.unsqueeze(0)) is None for x in images)}')
    launch_square_attack(forward_fn, images, labels)
    print(f'\nAttack done: {mode=}, {random_seed=}, {eps=}, {nqueries=}')
    print(f'{thresholds=}')
