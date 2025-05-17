import random

import numpy as np
import torch
import torchvision
from scipy import ndimage
import os
import random

import torchvision.transforms as T
from torch.utils.data import Dataset
from livenet.data_augment import RandomElasticShift, RandomRotateCrop


datasets_dir = f"{os.getenv('HOME')}/datasets/research"


class TransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform  = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = self.transform(x)
        return x, y


def permutate_sync(data1, data2, rng=None):
    assert len(data1) == len(data2)
    if rng is None:
        rng = np.random.default_rng(0)
    permutation = rng.permutation(len(data1))
    data1_perm = data1[permutation]
    data2_perm = data2[permutation]
    return data1_perm, data2_perm


def augment_vertical_flip(data, labels):
    data = torch.tensor(data, device="cpu")
    flipped = torch.flip(data, dims=[-1])
    flipped = flipped.numpy()
    augmented_x = np.concatenate((data, flipped), axis=0)
    augmented_y = np.concatenate((labels, labels), axis=0)
    augmented_x, augmented_y = permutate_sync(augmented_x, augmented_y)
    return augmented_x, augmented_y


def get_augmented_ten_rotations_and_shifts(data, labels, rotation_degrees: float, crop_pixels: int):
    augmented_x = get_ten_rotation_shifts(data, rotation_degrees, crop_pixels)
    augmented_y = np.concatenate(10 * [labels], axis=0)
    augmented_x, augmented_y = permutate_sync(augmented_x, augmented_y)
    return augmented_x, augmented_y


def get_augmented_eight_elastic_shifts(data, labels, shift:int):
    assert len(data) == len(labels)
    n = len(data)
    augmented_x = np.empty_like(data, shape=(8 * n, *data.shape[1:]))
    for i in range(n):
        img = data[i]
        augmented_x[i * 8 + 0] = _elastic_transform(img, (0, -shift))
        augmented_x[i * 8 + 1] = _elastic_transform(img, (0, shift))
        augmented_x[i * 8 + 2] = _elastic_transform(img, (-shift, 0))
        augmented_x[i * 8 + 3] = _elastic_transform(img, (shift, 0))
        augmented_x[i * 8 + 4] = _elastic_transform(img, (-shift, -shift))
        augmented_x[i * 8 + 5] = _elastic_transform(img, (-shift, shift))
        augmented_x[i * 8 + 6] = _elastic_transform(img, (shift, -shift))
        augmented_x[i * 8 + 7] = _elastic_transform(img, (shift, shift))
    augmented_y = np.repeat(labels, 8, axis=0)
    augmented_x, augmented_y = permutate_sync(augmented_x, augmented_y)
    return augmented_x, augmented_y


def get_ten_rotation_shifts(data: np.ndarray, rotation_degrees: float, crop_pixels: int):
    assert data.ndim == 4 # assumes input is nchw
    augmented = []

    for angle in [-rotation_degrees, rotation_degrees]:
        r = ndimage.rotate(data, angle, axes=(3, 2))
        r = r[:, :, 1:-1,1:-1]
        c = crop_pixels
        augmented.append(r[:, :, :-c, :-c])
        augmented.append(r[:, :, c:, :-c])
        augmented.append(r[:, :, c:, c:])
        augmented.append(r[:, :, :-c, c:])
        augmented.append(r[:, :, c // 2:-c // 2, c // 2:-c // 2])

    result = np.concatenate(augmented, axis=0)
    return result


def get_xor():
    xor_x = torch.Tensor([[0., 0.],
                          [0., 1.],
                          [1., 0.],
                          [1., 1.]])
    xor_y = torch.tensor([0, 1, 1, 0]).reshape(xor_x.shape[0], 1)
    return xor_x, xor_y


def get_odd_2():
    odd_x = torch.tensor([[0.], [1.], [2.]])
    odd_y = torch.tensor([[0], [1], [0]])
    return odd_x, odd_y


def get_pyramid():
    x = torch.Tensor([[0., 0, 0],
                      [0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0],
                      [1, 1, 1]])
    y = torch.tensor([[1], [0], [0], [0], [1]])
    return x, y


def get_linear2():
    x = torch.tensor([[1., 0],
                      [0, 1]
                      ])
    y = torch.tensor([[0], [1]])
    return x, y


def get_linear3():
    x = torch.tensor([[1, 0, 1],
                      [0, 1, 0.]
                      ])
    y = torch.tensor([[0], [1]])
    return x, y


def _get_mnist(train: bool):
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()
            , torchvision.transforms.Normalize([0.5], [0.5])
         ])
    dataset=torchvision.datasets.MNIST(datasets_dir, train=train,
                                    download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    whole = next(iter(loader))
    data = whole[0]
    labels = whole[1]
    labels = labels[:, None]
    return data, labels


def get_mnist_test():
    return _get_mnist(False)


def get_mnist_train():
    return _get_mnist(True)


def _get_cifar10(train: bool):
    dataset = torchvision.datasets.CIFAR10(datasets_dir, train=train, download=True, transform=torchvision.transforms.PILToTensor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    whole = next(iter(loader))
    data = whole[0]
    labels = whole[1]
    labels = labels[:, None]
    return data, labels


def get_cifar10_test():
    return _get_cifar10(False)


def get_cifar10_train():
    data, labels = _get_cifar10(True)
    return data, labels

# cifar10_normalization = torchvision.transforms.Normalize(127, 128
    # mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    # std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
# )

cifar10_normalization = torchvision.transforms.Lambda(lambda t: (t.to(torch.float32) - 127) / 128)

auto_aug_policy = [
    #(("Invert", 0.1, None), ("Contrast", 0.2, 6)),
    # (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
    (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
    # (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
    (("AutoContrast", 0.5, None), ("Equalize", 0.9, None)),
    (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
    (("Color", 0.4, 3), ("Brightness", 0.3, 7)),
    (("Sharpness", 0.3, 9), ("Brightness", 0.3, 9)),
    (("Equalize", 0.6, None), ("Equalize", 0.5, None)),
    (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
    # (("Color", 0.7, 7), ("TranslateX", 0.5, 8)),
    (("Equalize", 0.3, None), ("AutoContrast", 0.4, None)),
    (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)),
    (("Brightness", 0.3, 6), ("Color", 0.2, 8)),
    # (("Solarize", 0.5, 2), ("Invert", 0.0, None)),
    (("Equalize", 0.2, None), ("AutoContrast", 0.6, None)),
    (("Equalize", 0.2, None), ("Equalize", 0.6, None)),
    (("Color", 0.9, 9), ("Equalize", 0.6, None)),
    (("AutoContrast", 0.8, None), ("Solarize", 0.2, 8)),
    (("Brightness", 0.1, 3), ("Color", 0.7, 0)),
    (("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)),
    # (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
    # (("AutoContrast", 0.9, None), ("Solarize", 0.8, 3)),
    # (("Equalize", 0.8, None), ("Invert", 0.1, None)),
    # (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, None)),
]

auto_aug_policy1 = [
    (("Solarize", 0.5, 2), ("Invert", 0.0, None)),
]

auto_aug = T.AutoAugment()
auto_aug.policies = auto_aug_policy

def with_prob(f, probability):
    rng = random.Random(x=0)
    def f_prob(x):
        r = rng.random()
        if r < probability:
            return f(x)
        return x
    return f_prob


def func_chain(f1, f2):
    def chain_impl(x):
        return f1(f2(x))
    return chain_impl

elastic = RandomElasticShift(max_shift=4)
rot_crop = RandomRotateCrop(max_rotation=15)

cifar10_train_transform = torchvision.transforms.Compose(
    [
        # torchvision.transforms.RandomRotation(15),
        # torchvision.transforms.RandomAffine(12, (0.12, 0.12)),
        # torchvision.transforms.RandomCrop(32, padding=4, padding_mode="edge"),
        torchvision.transforms.RandomHorizontalFlip(),
        with_prob(func_chain(rot_crop, elastic), 1.5),
        auto_aug,
        cifar10_normalization
    ]
)

cifar10_test_transform = torchvision.transforms.Compose(
    [
        cifar10_normalization
    ]
)

def to_plain(x, y, downscale=1, to_odd=False, to_gray=False):
    if not isinstance(downscale, tuple):
        d = (downscale, downscale)
    else:
        d = downscale
    n0 = x.shape[2]
    n1 = x.shape[3]
    assert n0 % d[0] == 0 and n1 % d[1] == 0
    x = x.reshape(x.shape[0], x.shape[1], n0 // d[0], d[0], n1 // d[1], d[1])
    x = x.mean(axis=(3, 5), keepdims=False)
    if to_gray:
        x = x.mean(axis=1, keepdims=False)
    x = x.reshape(len(x), -1)
    if to_odd:
        y = y % 2
    return x, y


if __name__ == "__main__":
    t = get_cifar10_train()
    pass

