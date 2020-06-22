#!/usr/bin/env python3
from demo_superpoint import SuperPointFrontend
from data_loader import ImagePairLoader
import argparse
import torch
from torchvision import transforms
import numpy as np
import os
import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help="path to folder with image pairs")
    parser.add_argument('export', type=str, help="batch export input images and feature maps")
    parser.add_argument('-a', '--augmentation', action='store_true')
    parser.add_argument('--background', type=str, help="replacement for invalid pixels: either {random, mean} or a path to a folder with images")
    parser.add_argument('-s', '--seed', type=int, help="seed for deterministic augmentation")
    args = parser.parse_args()

    loader_conf = dict()
    if args.background:
        loader_conf['background'] = {
            'mode': args.background,
            'augmentation': {
                'colour': False,
                'translation': 0.2,
                'rotation': np.radians(45)
            }
        }
    if args.augmentation:
        loader_conf['augmentation'] = {
            'translation': 0.2,
            'rotation': np.radians(45)
        }

    data_path = args.data_path if args.data_path is not None else str()
    data_loader = ImagePairLoader(data_path, loader_conf, seed=args.seed)

    if not os.path.isdir(args.export):
        os.makedirs(args.export)

    spfe = SuperPointFrontend(weights_path='superpoint_v1.pth', nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=torch.cuda.is_available())

    for i_sample in range(len(data_loader)):
        I0, I1, map_xy, mask, _, _, _, _ = data_loader[i_sample]
        imgs = [I0, I1]
        fms = [None] * len(imgs)

        for i in range(2):
            # convert to grey scale
            img = cv2.cvtColor((imgs[i]*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)/255
            pts, desc, coarse_desc, heatmap = spfe.run(img)
            # move feature dim to end
            coarse_desc = np.moveaxis(coarse_desc[0], 0, -1)
            coarse_desc = cv2.resize(coarse_desc, tuple(np.roll(img.shape[:2], 1)))

            np.savez_compressed(os.path.join(args.export, str(i_sample*2+i)), I=img, F=coarse_desc)
