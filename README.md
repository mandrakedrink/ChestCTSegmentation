## CT Chest Segmentation
---
### Motivation 
Automatic segmentation of medical images is an important step to extract useful information that can help doctors make a diagnosis. Lung segmentation constitutes a critical procedure for any clinical-decision supporting system aimed to improve the early diagnosis and treatment of lung diseases. 

### Data
Available [here]() or [here](https://drive.google.com/drive/folders/1krhZD2R4QORhL_SiXNwqi1KRJ2s9zP-2?usp=sharing).
This dataset was be modified from `Lung segmentation dataset by Kónya et al., 2020 , https://www.kaggle.com/sandorkonya/ct-lung-heart-trachea-segmentation`
The code with  dataset  creation available here - [dataset_creation.ipynb](https://github.com/mandrakedrink/ChestCTSegmentation/blob/master/dataset_creation.ipynb)[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/drive/166TOgOsRvcblQK2j_HTB8CmVy5VGabas?usp=sharing)

More information about dataset can be read [here].

### Formulation of the problem:
Each pixel must be labeled “1” if it is part of one of the classes (**lungs**, **heart**, **trachea**), and “0” if not.

### Solution

The code with the solution is available here - The code with the solution is available here - [ct_chest_seg.ipynb](https://github.com/mandrakedrink/ChestCTSegmentation/blob/master/ct_chest_segmentation.ipynb)[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/drive/12MNwOSHp7JkVB3jkabqVXSTJoR4jZArm?usp=sharing)

### Results 

![](https://github.com/mandrakedrink/ChestCTSegmentation/blob/master/stats/result/svg/result3.svg)

<p>
 <img src="https://github.com/mandrakedrink/ChestCTSegmentation/blob/master/stats/result/svg/result1.svg" width="40%" height="40%">
 &emsp;&emsp;&emsp;
 <img src="https://github.com/mandrakedrink/ChestCTSegmentation/blob/master/stats/result/result-demov.gif" width="44%" height="44%">
</p>

----
[Video](https://youtu.be/HXTJRO2o3ys) with several examples.
