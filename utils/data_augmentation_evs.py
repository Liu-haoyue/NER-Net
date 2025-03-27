import torch
import torch.nn.functional as F
import torchvision.transforms
from math import sin, cos, pi
import numbers
import numpy as np
import random
from typing import Union

"""
    Data augmentation functions.

    There are some problems with torchvision data augmentation functions:
    1. they work only on PIL images, which means they cannot be applied to tensors with more than 3 channels,
       and they require a lot of conversion from Numpy -> PIL -> Tensor

    2. they do not provide access to the internal transformations (affine matrices) used, which prevent
       applying them for more complex tasks, such as transformation of an optic flow field (for which
       the inverse transformation must be known).

    For these reasons, we implement my own data augmentation functions
    (strongly inspired by torchvision transforms) that operate directly
    on Torch Tensor variables, and that allow to transform an optic flow field as well.
"""


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> torchvision.transforms.Compose([
        >>>     torchvision.transforms.CenterCrop(10),
        >>>     torchvision.transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms
        self.cord = None
        self.flip = None

    def __call__(self, x, is_flow=False):
        for t in self.transforms:
            x, flag, param = t(x, is_flow)
            if flag == 'RandomCrop' or flag == 'CenterCrop':
                self.cord = param
            elif flag == 'RandomFlip':
                self.flip = param
        return x, self.cord, self.flip

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomCrop(object):
    """Crop the tensor at a random location.
    """

    def __init__(self, size, preserve_mosaicing_pattern=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.preserve_mosaicing_pattern = preserve_mosaicing_pattern

    @staticmethod
    def get_params(x, output_size):
        w, h = x.shape[2], x.shape[1]
        th, tw = output_size
        if th > h or tw > w:
            raise Exception("Input size {}x{} is less than desired cropped \
                    size {}x{} - input tensor shape = {}".format(w,h,tw,th,x.shape))
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return i, j, th, tw

    def __call__(self, x, is_flow=False):
        """
            x: [C x H x W] Tensor to be rotated.
            is_flow: this parameter does not have any effect
        Returns:
            Tensor: Cropped tensor.
        """
        i, j, h, w = self.get_params(x, self.size)

        if self.preserve_mosaicing_pattern:
            # make sure that i and j are even, to preserve the mosaicing pattern
            if i % 2 == 1:
                i = i + 1
            if j % 2 == 1:
                j = j + 1

        return x[:, i:i + h, j:j + w], 'RandomCrop', [i, h, j, w]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomFlip(object):
    """
    Flip tensor along last two dims
    """

    def __init__(self, p_hflip=0.5, p_vflip=0.5):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    def __call__(self, x, is_flow=False):
        """
        :param x: [... x H x W] Tensor to be flipped.
        :param is_flow: if True, x is an [... x 2 x H x W] displacement field, which will also be transformed
        :return Tensor: Flipped tensor.
        """
        assert(len(x.shape) >= 2)
        if is_flow:
            assert(len(x.shape) >= 3)
            assert(x.shape[-3] == 2)

        dims = []
        if random.random() < self.p_hflip:  # horizontal flip 水平翻转(w)
            dims.append(-1)

        if random.random() < self.p_vflip:  # vertical flip 垂直翻转(h)
            dims.append(-2)

        if not dims:
            return x, 'RandomFlip', dims

        flipped = torch.flip(x, dims=dims)
        if is_flow:
            for d in dims:
                idx = -(d + 1)  # swap since flow is x, y
                flipped[..., idx, :, :] *= -1

        return flipped, 'RandomFlip', dims

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += '(p_flip={:.2f}'.format(self.p_hflip)
        format_string += ', p_vlip={:.2f})'.format(self.p_vflip)
        return format_string


class CenterCrop(object):
    """Center crop the tensor to a certain size.
    """

    def __init__(self, size, preserve_mosaicing_pattern=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.preserve_mosaicing_pattern = preserve_mosaicing_pattern

    def __call__(self, x, is_flow=False):
        """
            x: [C x H x W] Tensor to be rotated.
            is_flow: this parameter does not have any effect
        Returns:
            Tensor: Cropped tensor.
        """
        w, h = x.shape[2], x.shape[1]
        th, tw = self.size
        assert(th <= h)
        assert(tw <= w)
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        if self.preserve_mosaicing_pattern:
            # make sure that i and j are even, to preserve
            # the mosaicing pattern
            if i % 2 == 1:
                i = i + 1
            if j % 2 == 1:
                j = j + 1

        # return x[:, i:i + th, j:j + tw], 'CenterCrop', [i, th, j, tw]
        return x[:, i:i + th, j:j + tw], [i, th, j, tw], []

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


def normalize_image_sequence_(sequence, inverse_image, key='frame'):
    images = torch.stack([item[key] for item in sequence], dim=0)
    mini = np.percentile(torch.flatten(images), 1)
    maxi = np.percentile(torch.flatten(images), 99)
    images = (images - mini) / (maxi - mini + 1e-5)
    images = torch.clamp(images, 0, 1)

    for i, item in enumerate(sequence):
        if inverse_image:
            item[key] = 1 - images[i, ...]
        else:
            item[key] = images[i, ...]


def put_hot_pixels_in_voxel_(voxel, hot_pixel_range=1.0, hot_pixel_fraction=0.001):
    num_hot_pixels = int(hot_pixel_fraction * voxel.shape[-1] * voxel.shape[-2])
    x = torch.randint(0, voxel.shape[-1], (num_hot_pixels,))
    y = torch.randint(0, voxel.shape[-2], (num_hot_pixels,))
    for i in range(num_hot_pixels):
        voxel[..., :, y[i], x[i]] = random.uniform(-hot_pixel_range, hot_pixel_range)


def add_hot_pixels_to_sequence_(sequence, hot_pixel_std=1.0, max_hot_pixel_fraction=0.001):
    hot_pixel_fraction = random.uniform(0, max_hot_pixel_fraction)
    voxel = sequence[0]['events']
    num_hot_pixels = int(hot_pixel_fraction * voxel.shape[-1] * voxel.shape[-2])
    x = torch.randint(0, voxel.shape[-1], (num_hot_pixels,))
    y = torch.randint(0, voxel.shape[-2], (num_hot_pixels,))
    val = torch.randn(num_hot_pixels, dtype=voxel.dtype, device=voxel.device)
    val *= hot_pixel_std
    # TODO multiprocessing
    for item in sequence:
        for i in range(num_hot_pixels):
            item['events'][..., :, y[i], x[i]] += val[i]


def add_noise_to_voxel(voxel, noise_std=1.0, noise_fraction=0.1):
    noise = noise_std * torch.randn_like(voxel)  # mean = 0, std = noise_std
    if noise_fraction < 1.0:
        mask = torch.rand_like(voxel) >= noise_fraction
        noise.masked_fill_(mask, 0)
    return voxel + noise