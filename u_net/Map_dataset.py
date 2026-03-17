import os
import torch.utils.data as data
import cv2

from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

class MapDataset(data.Dataset):
    def __init__(self, root, cfg, image_set='train', transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform

        self.image_set = image_set
        self.cfg = cfg

        list_file = os.path.join(self.cfg.DATA_PATH, "{}.lst".format(image_set))

        self.images = []
        self.masks = []

        lines = open(list_file, 'rt')
        for line in lines:
            image_file, mask_file = line.strip().split(' ')
            self.images.append(os.path.join(root, image_file))
            self.masks.append(os.path.join(root, mask_file))

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        if self.cfg.NUM_CHANNELS == 3:
            img = Image.open(self.images[index]).convert('RGB')
        elif self.cfg.NUM_CHANNELS == 4:
            img = Image.open(self.images[index]).convert('RGBA')
        else:
            print("self.cfg.NUM_CHANNELS is wrong ", self.cfg.NUM_CHANNELS)
        # target = Image.open(self.masks[index])
        target = Image.fromarray(cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE))
        if self.transform is not None:
            img, target = self.transform(img, target)
        label = self.encode_target(target)
        label = np.asarray(label, dtype=np.int8)
        label_over = np.where(label >= self.cfg.NUM_CLASSES)
        label[label_over] = self.cfg.ignore_label
        label = np.asarray(label, dtype=np.uint8)

        return img, label


    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target_L(cls, mask, visible_mapping):
        """decode semantic mask to L(1channel) image"""
        # dst = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE))
        # mask = np.array(mask)
        for k, v in visible_mapping.items():
            mask[mask==k]=v
        return mask
        # return Image.fromarray(mask)        
