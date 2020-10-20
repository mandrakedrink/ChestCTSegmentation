### Dataset Description

CT Chest Segmentation Dataset.
This dataset was be modified from `Lung segmentation dataset by Kónya et al., 2020 , https://www.kaggle.com/sandorkonya/ct-lung-heart-trachea-segmentation`

The original nrrd files were re-saved in single tensor format with masks corresponding to labels: (**lungs**, **heart**, **trachea**) as numpy arrays using pickle.

Each tensor has the following shape: number of slices, width, height, number of classes, where the width and height number of slices are individual parameters of each tensor id, and number of classes = 3.

In addition, the data was re-saved as RGB images, where each image corresponds to one ID slice, and their mask-images have channels corresponding to three classes: (**lung**, **heart**, **trachea**).

<p>
  <img src="https://github.com/mandrakedrink/ChestCTSegmentation/blob/master/stats/data/demo.gif" width="35%" height="35%">
</p>



### Content
**The dataset contains**:

+ numpy_images_files.zip - images in numpy format.
+ numpy_masks_files.zip  - segmentation masks in numpy format.
+ images.zip - images in RGB format.
+ masks.zip - segmentation masks in RGB format.
+ train.csv - csv file with image names.


**Below is an example of what the data looks like:**

+ **.npy files can be readed like this:**

```
import pickle

with open(file_path, 'rb') as f:
    tensor = pickle.load(f)
```

+ **The images look like this:**
<p>
  <img src="https://github.com/mandrakedrink/ChestCTSegmentation/blob/master/stats/data/svg/data_example2.svg" width="60%" height="60%">
</p>

<p>
  <img src="https://github.com/mandrakedrink/ChestCTSegmentation/blob/master/stats/data/svg/data_example1.svg" width="60%" height="60%">
</p>


The code with  dataset  creation available here - [dataset_creation.ipynb](https://github.com/mandrakedrink/ChestCTSegmentation/blob/master/dataset_creation.ipynb)[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/drive/166TOgOsRvcblQK2j_HTB8CmVy5VGabas?usp=sharing)


---
