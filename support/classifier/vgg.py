import os

import torch
import torch.nn as nn

__all__ = [
    "VGG",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
]


class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        # CIFAR 10 (7, 7) to (1, 1)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            # nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class VGG_MIRROR(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG_MIRROR, self).__init__()
        self.features = features
        # CIFAR 10 (7, 7) to (1, 1)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            # nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class VGG_INVERSE(nn.Module):
    def __init__(self, list_tuple, num_classes=10, init_weights=True, on_cuda=True):
        super(VGG_INVERSE, self).__init__()
        (self.conv_layers, self.flag_list) = list_tuple

        # avg_pool: 1 x 1 --> 1 x 1
        # self.avgpool_inverse = nn.ConvTranspose2d(512, 512, 4, 2, 1)
        self.avgpool_inverse = nn.ConvTranspose2d(512, 512, 3, 1, 1) # FIXME

        self.linear_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(num_classes, 4096),
                          nn.ReLU(True),
                          nn.Dropout()),
            nn.Sequential(nn.Linear(4096, 4096),
                          nn.ReLU(True),
                          nn.Dropout()),
            nn.Linear(4096, 512 * 1 * 1)
        ])

        self.activate = nn.Tanh() # or nn.Sigmoid()?
        # self.activate = nn.Sigmoid()

        if init_weights:
            self._initialize_weights()

    def forward(self, x, neuron_outputs):
        base = 0
        for i, linear in enumerate(self.linear_layers):
            # print('linear model on cuda ?', next(linear.parameters()).is_cuda)
            x = linear(x * neuron_outputs[base + i])
        base = len(self.linear_layers)
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = self.avgpool_inverse(x)
        idx = 0
        for i, conv in enumerate(self.conv_layers):
            if self.flag_list[i]:
                x = conv(x * neuron_outputs[base + idx])
                idx += 1
            else:
                x = conv(x)
        x = self.activate(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_layers_inverse(cfg, batch_norm=False):
    layers = []
    flag_list = []
    out_ch = 512
    cfg_inverse = list(reversed([3] + cfg))
    for i in range(len(cfg_inverse)-1):
        in_ch = cfg_inverse[i]
        if in_ch == "M":
            # layers += [nn.MaxUnpool2d(kernel_size=2, stride=2)]
            layers += [nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)]
            flag_list.append(False)
        else:
            if cfg_inverse[i+1] == "M":
                out_ch = cfg_inverse[i+2]
            else:
                out_ch = cfg_inverse[i+1]
            convTranspose2d = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, padding=1)
            if batch_norm:
                layers += [nn.Sequential(convTranspose2d, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))]
            else:
                layers += [nn.Sequential(convTranspose2d, nn.ReLU(inplace=True))]
            flag_list.append(True)
    return (nn.ModuleList(layers), flag_list)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, device, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


def _vgg_inverse(arch, cfg, batch_norm, **kwargs):
    model = VGG_INVERSE(make_layers_inverse(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model

def vgg11_bn(pretrained=False, progress=True, device="cpu", **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11_bn", "A", True, pretrained, progress, device, **kwargs)


def vgg13_bn(pretrained=False, progress=True, device="cpu", **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13_bn", "B", True, pretrained, progress, device, **kwargs)


def vgg16_bn(pretrained=False, progress=True, device="cpu", **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16_bn", "D", True, pretrained, progress, device, **kwargs)


def vgg19_bn(pretrained=False, progress=True, device="cpu", **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19_bn", "E", True, pretrained, progress, device, **kwargs)


def vgg11_bn_inverse(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg_inverse("vgg11_bn", "A", True, **kwargs)


def vgg13_bn_inverse(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg_inverse("vgg13_bn", "B", True, **kwargs)


def vgg16_bn_inverse(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg_inverse("vgg16_bn", "D", True, **kwargs)


def vgg19_bn_inverse(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg_inverse("vgg19_bn", "E", True, **kwargs)
