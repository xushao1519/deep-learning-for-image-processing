import os

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from model import create_deep_pose_model
from utils import draw_keypoints
from datasets import WFLWDataset

def main():
    num_keypoints = 98
    img_path = "./test.jpg"
    weights_path = "./weights/model_weights_29.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.Resize([256, 256]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # load image
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    img_tensor: torch.Tensor = transform(img)
    # expand batch dimension
    img_tensor = img_tensor.unsqueeze_(0)

    # val_dataset = WFLWDataset(root="/home/wz/datasets/WFLW",
    #                           train=True,
    #                           transforms=transform)
    # img_tensor, label = val_dataset[0]
    # img_tensor = img_tensor.unsqueeze_(0)

    # create model
    model = create_deep_pose_model(num_keypoints=num_keypoints)

    # load model weights
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location="cpu")["model"])
    model.to(device)

    # prediction
    model.eval()
    with torch.inference_mode():
        res = torch.squeeze(model(img_tensor.to(device))).cpu().reshape([-1, 2])
        # print((res - label).pow(2).mean())
        draw_keypoints(np.array(img), rel_coordinate=res.numpy(), save_path="predict.jpg")


if __name__ == '__main__':
    main()
