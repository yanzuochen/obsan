# https://github.com/OwenSec/DeepDetector/blob/master/Test/CW/Test_CWL2_ImageNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

cross7x7 = torch.Tensor([
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
])
cross7x7 /= cross7x7.sum()

zero7x7 = torch.zeros_like(cross7x7)

def smoothed(xs, start, end, coefficient):
    newxs = torch.zeros_like(xs)
    # Fill the space before start and after end
    newxs[:, :, :start, :start] = xs[:, :, :start, :start]
    newxs[:, :, :start, end:] = xs[:, :, :start, end:]
    newxs[:, :, end:, :start] = xs[:, :, end:, :start]
    newxs[:, :, end:, end:] = xs[:, :, end:, end:]
    for xi, x in enumerate(xs):
        for row in range(start, end):
            for col in range(start, end):
                for chan in range(xs.shape[1]):
                    temp = x[chan][row][col]
                    for i in range(1,start+1):
                        temp += x[chan][row-i][col]
                        temp += x[chan][row+i][col]
                        temp += x[chan][row][col-i]
                        temp += x[chan][row][col+i]
                    newxs[xi][chan][row][col]= temp/coefficient
    return newxs

def quantized(xs, intervals):
    # Min, max for each image in xs
    mins = xs.view(xs.shape[0], -1).min(dim=1, keepdim=True).values
    maxs = xs.view(xs.shape[0], -1).max(dim=1, keepdim=True).values
    xs = (xs - mins) / (maxs - mins) * 255
    xs//=intervals
    xs*=intervals
    xs/=255.0
    xs = xs * (maxs - mins) + mins
    return xs

def entropies(xs):
    mins = xs.view(xs.shape[0], -1).min(dim=1, keepdim=True).values
    maxs = xs.view(xs.shape[0], -1).max(dim=1, keepdim=True).values
    xs = ((xs - mins) / (maxs - mins) * 255).to(torch.int16)
    nchans = xs.shape[1]
    img_size = xs.shape[2] * xs.shape[3]
    hs = []
    for x in xs:
        chan_hs = []
        for chan in range(nchans):
            pfreqs = torch.stack([
                torch.sum(x[chan] == i) for i in range(256)
            ]) / img_size
            chan_h = -torch.sum(torch.where(
                pfreqs > 0, pfreqs * torch.log2(pfreqs), torch.zeros_like(pfreqs)
            ))
            chan_hs.append(chan_h)
        hs.append(torch.sum(torch.stack(chan_hs)) / nchans)
    return torch.stack(hs)

def closer_blended(origxs, xsa, xsb):
    dista = torch.abs(origxs - xsa)
    distb = torch.abs(origxs - xsb)
    return torch.where(dista < distb, xsa, xsb)

class ANRProtectedModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def smoothedv2(self, x):
        smooth_conv_weight = torch.stack([
            torch.stack([cross7x7, zero7x7, zero7x7]),
            torch.stack([zero7x7, cross7x7, zero7x7]),
            torch.stack([zero7x7, zero7x7, cross7x7]),
        ])
        return F.conv2d(x, smooth_conv_weight, padding=3)

    def forward(self, x):
        logits = self.model(x)
        classes = torch.argmax(logits, dim=1)

        hs = entropies(x)
        quant_intervals = torch.where(
            hs < 4, 128,
            torch.where(
                hs < 5, 64, 43
            )
        )
        quantx = quantized(x, quant_intervals)
        # smoothed_qx = smoothed(quantx, 3, x.shape[2]-3, 13)
        smoothed_qx = self.smoothedv2(quantx)
        blended_sqx = closer_blended(x, quantx, smoothed_qx)
        processed_x = torch.where(hs < 5, quantx, blended_sqx)
        new_logits = self.model(processed_x)
        new_classes = torch.argmax(new_logits, dim=1)
        danger = (classes != new_classes).to(torch.float32)

        return logits, danger
