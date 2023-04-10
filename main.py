import sys

import cv2

from img_loader import ImageLoader
from img_treatment import ImagesTreatment
from grain_analysis import GrainAnalyser

path = sys.argv[1]
imgLoader = ImageLoader(path)
imgLoader.load()

grains_imgs = []

for image in imgLoader.images:
    imgTreatment = ImagesTreatment(image)
    grains_imgs += imgTreatment.final_segmentation()
    for grain in grains_imgs:
        cv2.imshow('grain', grain)
        cv2.waitKey(0)



grain_analyser = GrainAnalyser(grains_imgs)
df = grain_analyser.compute_df()
df.to_csv('grains_data.csv')
print(df)


