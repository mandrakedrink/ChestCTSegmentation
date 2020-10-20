import os

import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from albumentations.pytorch import ToTensor, ToTensorV2 
from albumentations import (HorizontalFlip,
                            VerticalFlip,
                            Normalize,
                            Compose)


class LungsDataset(Dataset):
    def __init__(self, 
                 imgs_dir: str,
                 masks_dir:str,
                 df: pd.DataFrame,
                 phase: str):
        """Initialization."""
        self.root_imgs_dir = imgs_dir
        self.root_masks_dir = masks_dir
        self.df = df
        self.augmentations = get_augmentations(phase)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "ImageId"]
        mask_name = self.df.loc[idx, "MaskId"]
        img_path = os.path.join(self.root_imgs_dir, img_name)
        mask_path = os.path.join(self.root_masks_dir, mask_name)
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        mask[mask < 240] = 0    # remove artifacts
        mask[mask > 0] = 1

        augmented = self.augmentations(image=img, 
                                       mask=mask.astype(np.float32))
        img = augmented['image']
        mask = augmented['mask'].permute(2, 0, 1)

        return img, mask


def get_augmentations(phase,
                   mean: tuple = (0.485, 0.456, 0.406),
                   std: tuple = (0.229, 0.224, 0.225),):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                VerticalFlip(p=0.5), 
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            #ToTensor(num_classes=3, sigmoid=False),
            ToTensorV2(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


def get_dataloader(
    imgs_dir: str,
    masks_dir: str,
    path_to_csv: str,
    phase: str,
    batch_size: int = 8,
    num_workers: int = 6,
    test_size: float = 0.2,
):
    '''Returns: dataloader for the model training'''
    df = pd.read_csv(path_to_csv)
    

    train_df, val_df = train_test_split(df, 
                                          test_size=test_size, 
                                          random_state=69)
    train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    df = train_df if phase == "train" else val_df
    image_dataset = LungsDataset(imgs_dir, masks_dir, df, phase)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,   
    )

    return dataloader
