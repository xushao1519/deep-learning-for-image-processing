import os
from typing import List

import cv2
import torch
import torch.utils.data as data
import numpy as np


class WFLWDataset(data.Dataset):
    """
    https://wywu.github.io/projects/LAB/WFLW.html

    dataset structure:

    ├── WFLW_annotations
    │   ├── list_98pt_rect_attr_train_test
    │   └── list_98pt_test
    └── WFLW_images
        ├── 0--Parade
        ├── 1--Handshaking
        ├── 10--People_Marching
        ├── 11--Meeting
        ├── 12--Group
        └── ......
    """
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transforms=None):
        super().__init__()
        self.img_root = os.path.join(root, "WFLW_images")
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
        ana_txt_name = "list_98pt_rect_attr_train.txt" if train else "list_98pt_rect_attr_test.txt"
        self.anno_path = os.path.join(root, "WFLW_annotations", "list_98pt_rect_attr_train_test", ana_txt_name)
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)

        self.transforms = transforms
        self.keypoints: List[torch.Tensor] = []
        self.face_rects: List[List[int]] = []
        self.img_paths: List[str] = []
        with open(self.anno_path, "rt") as f:
            for line in f.readlines():
                if not line.strip():
                    continue

                split_list = line.strip().split(" ")
                keypoint_ = self.get_98_points(split_list)
                keypoint = torch.as_tensor(keypoint_, dtype=torch.float32).reshape((-1, 2))
                face_rect = list(map(int, split_list[196: 196 + 4]))  # xmin, ymin, xmax, ymax
                img_name = split_list[-1]
                # absolute coordinate to relative coordinate
                keypoint[:, 0] -= face_rect[0]
                keypoint[:, 1] -= face_rect[1]
                # normalize coordinate to 0~1
                keypoint[:, 0] /= face_rect[2] - face_rect[0]
                keypoint[:, 1] /= face_rect[3] - face_rect[1]

                self.keypoints.append(keypoint)
                self.face_rects.append(face_rect)
                self.img_paths.append(os.path.join(self.img_root, img_name))

    @staticmethod
    def get_5_points(keypoints: List[str]) -> List[float]:
        five_num = [76, 82, 54, 96, 97]
        five_keypoint = []
        for i in five_num:
            five_keypoint.append(keypoints[i * 2])
            five_keypoint.append(keypoints[i * 2 + 1])
        return list(map(float, five_keypoint))

    @staticmethod
    def get_98_points(keypoints: List[str]) -> List[float]:
        return list(map(float, keypoints[:196]))

    def __getitem__(self, idx: int):
        img_bgr = cv2.imread(self.img_paths[idx], flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        xmin, ymin, xmax, ymax = self.face_rects[idx]
        crop_img = img[ymin:ymax, xmin:xmax]
        keypoint = self.keypoints[idx]
        mask = (keypoint > 0).sum(dim=1) == 2  # > 0 and < 1

        if self.transforms is not None:
            crop_img, keypoint = self.transforms(crop_img, keypoint)

        return crop_img, keypoint, mask

    def __len__(self):
        return len(self.keypoints)


if __name__ == '__main__':
    train_dataset = WFLWDataset("/home/wz/datasets/WFLW", train=True)
    print(len(train_dataset))

    eval_dataset = WFLWDataset("/home/wz/datasets/WFLW", train=False)
    print(len(eval_dataset))

    from utils import draw_keypoints
    img, keypoint = train_dataset[0]
    draw_keypoints(np.array(img), keypoint.numpy(), "test_plot.jpg")
