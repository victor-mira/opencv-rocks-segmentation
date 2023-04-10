import os
import cv2

class ImageLoader():
    def __init__(self, path):
        self.path = path
        self.images_names = os.listdir(path)
        self.images = []

    def load(self):
        for imgName in self.images_names:
            self.images.append(cv2.imread(self.path + imgName))

