import os
import cv2
import torch
import numpy as np
from f3net_model import F3Net
from utils.chargrid_converter import char_grid


f3net = F3Net()


def image_to_tensor(
        input_img_path
):
    mean = np.array([[[124.55, 118.90, 102.94]]])
    std = np.array([[[56.77, 55.97, 57.50]]])

    image = cv2.imread(input_img_path)[:, :, ::-1].astype(np.float32)
    shape = image.shape
    image = (image - mean) / std
    W = 352
    H = 352

    image = cv2.resize(
        image, dsize=(W, H),
        interpolation=cv2.INTER_LINEAR
    )

    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1)

    return image, shape


def predict(
        image,
        shape
):
    F3Net.cuda()
    with torch.no_grad():
        image = image.cuda().float()
        shape_mask = (shape[0], shape[1])
        out1u, out2u, out2r, out3r, out4r, out5r = F3Net(image[None, ...], shape_mask)

        out = out2u
        pred = (torch.sigmoid(out[0, 0]) * 255).cpu().detach().numpy()
        cv2.imwrite('/content/AL173(1).png', np.round(pred))


