import contextlib
import os
import numpy as np
from PIL import Image
import json

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from tvm.contrib.download import download_testdata

CIFAR_label_list = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

class CIFAR10Dataset(Dataset):
    def __init__(self,
                 image_size,
                 normalize=True,
                 image_dir='/dev/shm/deployed-datasets/cifar-10-png/',
                 split='train'):
        super(CIFAR10Dataset).__init__()
        # self.image_dir = image_dir + ('train/' if split == 'train' else 'test/')
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ] + (
            [transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2471, 0.2435, 0.2616))] if normalize else []
        ))
        # self.norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        self.image_list = []
        self.cat_list = sorted(os.listdir(self.image_dir))
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

class CIFAR10MonochromeDataset(CIFAR10Dataset):
    def __init__(self, image_size, *args, **kwargs):
        super().__init__(image_size, *args, **kwargs)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

class ImageNetDataset(Dataset):
    def __init__(self,
                 image_size,
                 image_dir='/dev/shm/deployed-datasets/CLS-LOC/',
                 label2index_file='/dev/shm/deployed-datasets/CLS-LOC/ImageNetLabel2Index.json',
                 split='val'):
        super(ImageNetDataset).__init__()
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.image_list = []

        with open(label2index_file, 'r') as f:
            self.label2index = json.load(f)

        self.cat_list = sorted(os.listdir(self.image_dir))

        for cat in self.cat_list:
            name_list = sorted(os.listdir(self.image_dir + cat))
            self.image_list += [self.image_dir + cat + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2] # label name
        index = self.label2index[label]
        index = torch.LongTensor([index]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index

class CelebADataset(Dataset):
    def __init__(self,
                 image_size,
                 image_dir='/dev/shm/deployed-datasets/celeba_crop128/',
                 split='train'):
        super(CelebADataset).__init__()
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))  # CIFAR10
        ])
        self.image_list = sorted(os.listdir(self.image_dir))
        print('Total %d data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image = Image.open(self.image_dir + image_name)
        image = self.transform(image)
        return image, -1

class ChestDataset(Dataset):
    def __init__(self,
                 image_size,
                 image_dir='/dev/shm/deployed-datasets/ChestX-jpg128-split/',
                 split='train'):
        super(ChestDataset).__init__()
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))  # CIFAR10
        ])
        self.image_list = sorted(os.listdir(self.image_dir))
        print('Total %d data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image = Image.open(self.image_dir + image_name)
        image = self.transform(image)
        return image, -1

class BrokenImageDataset(Dataset):
    def __init__(self, image_size, image_dir) -> None:
        super().__init__()
        self.image_size = image_size
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        self.image_list = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image = Image.open(f'{self.image_dir}/{image_name}')
        image = self.transform(image)
        # image = torch.zeros(3, self.image_size, self.image_size)
        return image, -1

def MNISTDataset(colourise, image_size, split='train'):
    return datasets.MNIST(
        root='/dev/shm/deployed-datasets',
        train=(split == 'train'),
        download=True,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ] + ([
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ] if colourise else []))
    )

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

class RandDataset(Dataset):
    def __init__(self, image_size, nimages=10000) -> None:
        super().__init__()
        self.image_size = image_size
        self.nimages = nimages
        with temp_seed(42):
            size = (nimages, 3, image_size, image_size)
            r = np.random.normal(size=size)
            r *= np.random.rand(*size) * 100
            r += np.random.rand(*size) * 200 - 100
            self.images = torch.tensor(r, dtype=torch.float32)

    def __len__(self):
        return self.nimages

    def __getitem__(self, index):
        return self.images[index], -1

def CIFAR10SubDataset(start, end, image_size, split='train'):
    return Subset(CIFAR10Dataset(image_size, split=split), range(start, end))

def gen_dl_broken_dataset(image_size):
    assert image_size == 32
    imgs = torch.load(f'/dev/shm/deployed-datasets/dl-broken-cifar10/dl-broken-resnet50-CIFAR10-train.pt')
    imgs = torch.tensor(np.concatenate(imgs, 0))
    return TensorDataset(imgs, torch.zeros(imgs.shape[0]))

benign_datasets = {
    'CIFAR10': CIFAR10Dataset,
    'ImageNet': ImageNetDataset,
    'MNIST': lambda *args, **kwargs: MNISTDataset(False, *args, **kwargs),
    'CIFAR10_2': lambda *args, **kwargs: CIFAR10SubDataset(0, 2*5000, *args, **kwargs),
    'CIFAR10RAW': lambda *args, **kwargs: CIFAR10Dataset(*args, **kwargs, normalize=False),
}

undef_datasets = {
    'CelebA': CelebADataset,
    'Chest': ChestDataset,
    'CIFAR10UD': lambda *args, **kwargs: CIFAR10SubDataset(2*5000, 3*5000, *args, **kwargs),
    'MNISTC': lambda *args, **kwargs: MNISTDataset(True, *args, **kwargs),
    'CIFAR10M': CIFAR10MonochromeDataset,
    'CIFAR10B': lambda image_size: BrokenImageDataset(image_size, '/dev/shm/deployed-datasets/broken-cifar10-white'),
    'ImageNetB': lambda image_size: BrokenImageDataset(image_size, '/dev/shm/deployed-datasets/broken-imagenet'),
    'rand': RandDataset,
    'dl': gen_dl_broken_dataset,
}

def make_loader(dataset, batch_size, num_workers=4, size_limit=0, dataset_handler=None, shuffle=False):
    assert sum(1 for x in [size_limit, dataset_handler] if x) <= 1
    if dataset_handler:
        dataset = dataset_handler(dataset)
    elif size_limit and len(dataset) > size_limit:
        dataset = Subset(dataset, range(size_limit))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def get_benign_loader(dataset_name, image_size, split, batch_size, **kwargs):
    return make_loader(benign_datasets[dataset_name](image_size, split=split), batch_size, **kwargs)

def get_undef_loader(dataset_name, image_size, batch_size=1, size_limit=10000, **kwargs):
    dataset = undef_datasets[dataset_name](image_size)
    return make_loader(dataset, batch_size, size_limit=size_limit, **kwargs)

def get_sampling_benign_loader(dataset_name, image_size, split, batch_size, frac_per_class, start_frac=0., **kwargs):
    if dataset_name.startswith('CIFAR10'):
        nclasses, nimgs_per_class = 10, (1000 if split == 'test' else 5000)
        nchosen_per_class = int(frac_per_class * nimgs_per_class)
    else:
        assert False
    ds_handler = lambda ds: Subset(
        ds,
        range(
            int(nimgs_per_class * start_frac),
            nclasses * nimgs_per_class,
            nimgs_per_class // nchosen_per_class
        )
    )
    data_loader = get_benign_loader(dataset_name, image_size, split, batch_size, dataset_handler=ds_handler, **kwargs)
    return data_loader

def get_ae_loader(model, dataset, batch_size, split='train', alg='PGD', size_limit=11000, **kwargs):
    assert alg in ['PGD', 'FGSM', 'CW', 'BIM', 'DeepFool']
    if model.startswith('Q'):
        model = model[1:]
    name = '%s-%s-%s-%s.pt' % (alg, model, dataset, split)

    adv_data = torch.load('/dev/shm/deployed-datasets/adversarial_examples-01/' + name,
                          map_location=torch.device('cpu'))

    # Deal with version differences
    if isinstance(adv_data, dict):
        adv_images = adv_data['adv_inputs']
        labels = adv_data['labels']
    else:
        adv_images, labels = adv_data

    adv_data = TensorDataset(adv_images, labels)
    return make_loader(adv_data, batch_size, size_limit=size_limit, **kwargs)

def get_labels(dataset: str):
    assert dataset in {'ImageNet', 'MNIST', 'CIFAR10', 'CIFAR10_2'}
    if dataset == 'ImageNet':
        labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
        labels_path = download_testdata(labels_url, "synset.txt", module="data")
        with open(labels_path, "r") as f:
            return [l.rstrip() for l in f]
    elif dataset == 'MNIST':
        return [str(i) for i in range(10)]
    elif dataset == 'CIFAR10':
        return CIFAR_label_list.copy()
    else:
        return CIFAR_label_list[:2].copy()
