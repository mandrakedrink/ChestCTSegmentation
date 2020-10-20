import numpy as np
import torch

import os
import io
import base64

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import time
from IPython.display import clear_output
from IPython.display import HTML


def show_data_augmentations(img_tensor: torch.Tensor,
                            mask_tensor: torch.Tensor,
                            mean: tuple = (0.485, 0.456, 0.406),
                            std: tuple = (0.229, 0.224, 0.225),
                            labels: list=["image", "lung", "heart", "trachea"]):
    
    img = img_tensor.numpy().transpose(1, 2, 0)
    img = (img * std + mean).astype("float32")
    img = np.clip(img, 0, 1)
    mask = mask_tensor.numpy().transpose(1, 2, 0)
    data_to_plot = [img, *[mask[:,:, i] for i in range(3)]]

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 8))
    for i, ax in enumerate(axes):
        ax.imshow(data_to_plot[i])
        ax.set_title(labels[i])

    plt.show()
    
    
def show_video(video_path: str):
    """
    show video in jupyter notebook, agent interaction in environment.
    Takes - path to video file.
    Returns - html video player in jupyter notebook.
    """  
    video = io.open(video_path, 'r+b').read()
    encoded = base64.b64encode(video)

    return HTML(data='''<video alt="test" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4" /> </video>'''
    .format(encoded.decode('ascii')))
  
