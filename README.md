
![Release](https://img.shields.io/badge/Release-v1.0-blueviolet)
![Language](https://img.shields.io/badge/Language-Python-0052cf)
![Libraries](https://img.shields.io/badge/Libraries-OpenCV_Numpy_Matplotlib_Panda-20d645)

# Rock Segmentation with OpenCV
### :warning: This was a School project with a deadline, thus it is not finished at the moment :warning:

This project allow to separate sets of rocks observed by microscope into individual image to analyse them.

## How to run
Run main.py in a console with the path to rock images folder in parameters
```
>>> py main.py "/my-folder/"
```
It will then save all the inviduals images in the "Grains" Folder and display means of Bleu, Green and Red for each grain. These values are stored in the grain_data.csv file aswell

## Process
To make a good segmentation we start by getting the peak local maximums, then we merges local max to closes to each other to avoid sur segmentation. 

| ![image](https://github.com/victor-mira/opencv-rocks-segmentation/assets/58742508/c80c491c-b37b-4fcc-9957-bbfaaf751c1e) | ![image](https://github.com/victor-mira/opencv-rocks-segmentation/assets/58742508/40ed0b02-be42-48f7-b02d-e2a28aebe5d5) |
| :-------------: |:-------------:|
| *Before merging* | *After merging* |

We uses a watershed algorithm on the tresholded images to create a mask for each grain and check the if the size of the grain match our expectation with the contour of the mask. We can now apply the mask to the original image and crop it before saving it.

| ![image](https://github.com/victor-mira/opencv-rocks-segmentation/assets/58742508/b2cf5bad-9ce9-4cb5-a59e-da6132a83b60) | ![image](https://github.com/victor-mira/opencv-rocks-segmentation/assets/58742508/3740f51a-71f6-436b-aef4-c90a00d90f2a) |
| :-------------: |:-------------:|
| *One grain mask* | *Cropped grain image* |

Once each grain from each image have been proccessed we can then analyse and store the value into a Panda datafile.
