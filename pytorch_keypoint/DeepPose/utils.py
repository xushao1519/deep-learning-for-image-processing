import cv2
import numpy as np


def draw_keypoints(img: np.ndarray, rel_coordinate: np.ndarray, save_path: str):
    h, w, c = img.shape
    coordinate = rel_coordinate.copy()
    coordinate[:, 0] *= w
    coordinate[:, 1] *= h
    coordinate = coordinate.astype(np.int64).tolist()

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for x, y in coordinate:
        cv2.circle(img_bgr, center=(x, y), radius=3, color=(255, 0, 0), thickness=-1)

    cv2.imwrite(save_path, img_bgr)
