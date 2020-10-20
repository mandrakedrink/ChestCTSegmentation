import os
import io
import base64

import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from albumentations import Normalize

import time
from IPython.display import clear_output
from IPython.display import HTML

from loss_metric import dice_coef_metric_per_classes, jaccard_coef_metric_per_classes

def get_one_slice_data(img_name: str,
                       mask_name: str,
                       root_imgs_path: str = "images/",
                       root_masks_path: str = "masks/",) -> np.ndarray:

    img_path = os.path.join('images/', img_name)
    mask_path = os.path.join('masks/', mask_name)
    one_slice_img = cv2.imread(img_path)#[:,:,0] uncomment for grayscale
    one_slice_mask = cv2.imread(mask_path)
    one_slice_mask[one_slice_mask < 240] = 0  # remove artifacts
    one_slice_mask[one_slice_mask >= 240] = 255

    return one_slice_img, one_slice_mask


def get_id_predictions(net: nn.Module,
                       ct_scan_id_df: pd.DataFrame,
                       root_imgs_dir: str,
                       treshold: float = 0.3) -> list:

    """
    Factory for getting predictions and storing them and images in lists as uint8 images.
    Params:
        net: model for prediction.
        ct_scan_id_df: df with unique patient id.
        root_imgs_dir: root path for images.
        treshold: threshold for probabilities.
    """
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    images = []
    predictions = []
    net.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    with torch.no_grad():
        for idx in range(len(ct_scan_id_df)):
            img_name = ct_scan_id_df.loc[idx, "ImageId"]
            path = os.path.join(root_imgs_dir, img_name)

            img_ = cv2.imread(path)
    
            img = Normalize().apply(img_)
            tensor = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0)
            prediction = net.forward(tensor.to(device))
            prediction = prediction.cpu().detach().numpy()
            prediction = prediction.squeeze(0).transpose(1, 2, 0)
            prediction = sigmoid(prediction)
            prediction = (prediction >= treshold).astype(np.float32)

            predictions.append((prediction * 255).astype("uint8"))
            images.append(img_)

    return images, predictions


# Save image in original resolution
# helpful link - https://stackoverflow.com/questions/34768717/matplotlib-unable-to-save-image-in-same-resolution-as-original-image

def get_overlaid_masks_on_image(
                one_slice_image: np.ndarray,
                one_slice_mask: np.ndarray, 
                w: float = 512,
                h: float = 512, 
                dpi: float = 100,
                write: bool = False,
                path_to_save: str = '/content/',
                name_to_save: str = 'img_name'):
    """overlap masks on image and save this as a new image."""

    path_to_save_ = os.path.join(path_to_save, name_to_save)
    lung, heart, trachea = [one_slice_mask[:, :, i] for i in range(3)]
    figsize = (w / dpi), (h / dpi)
    fig = plt.figure(figsize=(figsize))
    fig.add_axes([0, 0, 1, 1])

    # image
    plt.imshow(one_slice_image, cmap="bone")

    # overlaying segmentation masks
    plt.imshow(np.ma.masked_where(lung == False, lung),
            cmap='cool', alpha=0.3)
    plt.imshow(np.ma.masked_where(heart == False, heart),
            cmap='autumn', alpha=0.3)
    plt.imshow(np.ma.masked_where(trachea == False, trachea),
               cmap='autumn_r', alpha=0.3) 

    plt.axis('off')
    fig.savefig(f"{path_to_save_}.png",bbox_inches='tight', 
                pad_inches=0.0, dpi=dpi,  format="png")
    if write:
        plt.close()
    else:
        plt.show()
        
        
def get_overlaid_masks_on_full_ctscan(ct_scan_id_df: pd.DataFrame, path_to_save: str):
    """
    Creating images with overlaid masks on each slice of CT scan.
    Params:
         ct_scan_id_df: df with unique patient id.
         path_to_save: path to save images.
    """
    num_slice = len(ct_scan_id_df)
    for slice_ in range(num_slice):
        img_name = ct_scan_id_df.loc[slice_, "ImageId"]
        mask_name = ct_scan_id_df.loc[slice_, "MaskId"]
        one_slice_img, one_slice_mask = get_one_slice_data(img_name, mask_name)
        get_overlaid_masks_on_image(one_slice_img,
                                one_slice_mask,
                                write=True, 
                                path_to_save=path_to_save,
                                name_to_save=str(slice_)
                                )

def create_video(path_to_imgs: str, video_name: str, framerate: int):
    """
    Create video from images.
    Params:
        path_to_imgs: path to dir with images.
        video_name: name for saving video.
        framerate: num frames per sec in video.
    """
    img_names = sorted(os.listdir(path_to_imgs), key=lambda x: int(x[:-4]))  # img_name must be numbers
    img_path = os.path.join(path_to_imgs, img_names[0])
    frame_width, frame_height, _ = cv2.imread(img_path).shape
    fourc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(video_name + ".mp4", 
                            fourc, 
                            framerate, 
                            (frame_width, frame_height))

    for img_name in img_names:
        img_path = os.path.join(path_to_imgs, img_name)
        image = cv2.imread(img_path)
        video.write(image)
            
    cv2.destroyAllWindows()
    video.release()

    
def compute_scores_per_classes(model,
                               dataloader,
                               classes):
    """
    Compute Dice and Jaccard coefficients for each class.
    Params:
        model: neural net for make predictions.
        dataloader: dataset object to load data from.
        classes: list with classes.
        Returns: dictionaries with dice and jaccard coefficients for each class for each slice.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dice_scores_per_classes = {key: list() for key in classes}
    iou_scores_per_classes = {key: list() for key in classes}

    with torch.no_grad():
        for i, (imgs, targets) in enumerate(dataloader):
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            logits = logits.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            
            dice_scores = dice_coef_metric_per_classes(logits, targets)
            iou_scores = jaccard_coef_metric_per_classes(logits, targets)

            for key in dice_scores.keys():
                dice_scores_per_classes[key].extend(dice_scores[key])

            for key in iou_scores.keys():
                iou_scores_per_classes[key].extend(iou_scores[key])

    return dice_scores_per_classes, iou_scores_per_classes
