import random

import cv2
import torch
import numpy as np

from wflw_horizontal_flip_indices import wflw_flip_indices_dict


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, keypoint):
        for t in self.transforms:
            image, keypoint = t(image, keypoint)
        return image, keypoint


class Resize(object):
    def __init__(self, h: int, w: int):
        self.h = h
        self.w = w

    def __call__(self, image: np.ndarray, keypoint):
        image = cv2.resize(image, dsize=(self.w, self.h), fx=0, fy=0,
                           interpolation=cv2.INTER_LINEAR)

        return image, keypoint


class MatToTensor(object):
    """将opencv图像转为Tensor, HWC2CHW, 并缩放数值至0~1"""
    def __call__(self, image, keypoint):
        image = torch.from_numpy(image).permute((2, 0, 1))
        image = image.to(torch.float32) / 255.
        return image, keypoint


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = torch.as_tensor(mean, dtype=torch.float32).reshape((3, 1, 1))
        self.std = torch.as_tensor(std, dtype=torch.float32).reshape((3, 1, 1))

    def __call__(self, image: torch.Tensor, keypoint):
        image.sub_(self.mean).div_(self.std)
        return image, keypoint


class RandomHorizontalFlip(object):
    """随机对输入图片进行水平翻转"""
    def __init__(self, p: float = 0.5):
        self.p = p
        self.wflw_flip_ids = list(wflw_flip_indices_dict.values())

    def __call__(self, image: np.ndarray, keypoint: torch.Tensor):
        if random.random() < self.p:
            # [h, w, c]
            image = np.ascontiguousarray(np.flip(image, axis=[1]))

            # [k, 2]
            keypoint = keypoint[self.wflw_flip_ids]
            keypoint[:, 0] = 1. - keypoint[:, 0]

        return image, keypoint
