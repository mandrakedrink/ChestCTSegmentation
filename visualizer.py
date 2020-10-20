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
  
  
def get_color_info(classes: list = ['lung', 'heart', 'trachea']):

    def get_color(cmap):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            (cmap.name, 0.0, 0.0,),
            cmap(np.linspace(0.0, 0.0, 100))
            )
        return new_cmap

    colormaps = [plt.get_cmap(cmap_name) for cmap_name in 
                ['cool', 'autumn', 'autumn_r']
                ]
    arr = np.linspace(0, 50, 100).reshape((10, 10))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        cmap = get_color(colormaps[i])
        ax[i].axis('off')
        ax[i].imshow(arr, cmap=cmap)
        ax[i].set_title(classes[i], fontsize=15)

    fig.suptitle("Color definition", fontsize=20, y=0.99)
    fig.savefig(f"color_definition.png", bbox_inches='tight', 
                pad_inches=0.2, dpi=100, format="png")
    
    plt.savefig(f"color_definition.svg", bbox_inches='tight',
                pad_inches=0.2, dpi=100, format="svg")
    plt.show()
