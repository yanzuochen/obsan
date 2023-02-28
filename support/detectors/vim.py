#! /usr/bin/env python3
# Reference: https://gitlab.com/kkirchheim/pytorch-ood/-/blob/dev/src/pytorch_ood/detector/vim.py

from typing import Tuple
import argparse

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from tqdm import tqdm
from sklearn.covariance import EmpiricalCovariance

class ViMProtectedModule(nn.Module):
    def __init__(self, model, model_fc, R, alpha):
        super().__init__()
        self.model = model
        self.fc = model_fc

        self.R = R
        self.alpha = alpha

        with torch.no_grad():
            fc_w, fc_b = [p.cpu().numpy() for p in model_fc.parameters()]
            self.u = -np.matmul(pinv(fc_w), fc_b)

    def run_model(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the input of the fc layer and the output of the model."""
        fc_input = None
        def hook(module, input, output):
            nonlocal fc_input
            fc_input = input[0]
        handle = self.fc.register_forward_hook(hook)
        output = self.model(x)
        handle.remove()
        return fc_input, output

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note that this implementation is limited to operators that can be
        # traced by PyTorch and compiled by TVM.

        def norm(x):
            return torch.sqrt(torch.sum(x ** 2, dim=-1))

        maybe_to_torch = lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        self.R, self.u = [maybe_to_torch(p) for p in [self.R, self.u]]

        with torch.no_grad():
            features, logits = self.run_model(x)
            features -= self.u

        # calculate residual
        x_p_t = (self.R @ self.R.t() @ features.t()).t()
        vlogit = norm(x_p_t) * self.alpha
        energy = -torch.logsumexp(logits, dim=-1)
        sus_score = vlogit + energy

        return logits, sus_score

    def build(self, data_loader, outfile=None, device="cpu"):
        # extract features
        with torch.no_grad():
            features_l, logits_l = [], []

            for x, y in tqdm(data_loader):
                features, logits = self.run_model(x.to(device))
                features_l.append(features.cpu())
                logits_l.append(logits.cpu())

        features = torch.concat(features_l).numpy()
        logits = torch.concat(logits_l).numpy()
        D = 1000 if features.shape[1] > 1500 else 512

        # calculate eigenvectors of the covariance matrix
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(features - self.u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)

        # select largest eigenvectors to get the principal subspace
        largest_eigvals_idx = np.argsort(eig_vals * -1)[D:]
        self.R = np.ascontiguousarray((eigen_vectors.T[largest_eigvals_idx]).T)

        # calculate residual
        x_p_t = (self.R @ self.R.T @ (features - self.u).T).T
        vlogits = norm(x_p_t, axis=-1)
        self.alpha = logits.max(axis=-1).mean() / vlogits.mean()

        if outfile:
            torch.save([self.R, self.alpha], outfile)

        return self.R, self.alpha

if __name__ == "__main__":

    import sys, os
    sys.path.insert(1, os.path.join(sys.path[0], '../..'))
    import modman
    import dataman

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, choices=['resnet50', 'googlenet', 'densenet121'])
    parser.add_argument("outfile", type=str, help="Path to the build output file.")
    parser.add_argument("--dataset", default='CIFAR10', choices=['CIFAR10'])
    parser.add_argument("--device", type=str, default="cpu", help="Device to use.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    args = parser.parse_args()

    # load model
    model, fc = None, None
    if args.model_name in ['resnet50', 'googlenet', 'densenet121']:
        model = modman.get_torch_mod(args.model_name, args.dataset)
        fc = getattr(model, 'fc', getattr(model, 'classifier', None))
        assert isinstance(fc, nn.Linear)
    else:
        raise NotImplementedError

    # load data
    train_loader = None
    if args.dataset == "CIFAR10":
        train_loader = dataman.get_benign_loader(args.dataset, 32, 'train', args.batch_size)
    else:
        raise NotImplementedError

    detector = ViMProtectedModule(model, fc, None, None)

    detector.build(train_loader, outfile=args.outfile, device=args.device)
