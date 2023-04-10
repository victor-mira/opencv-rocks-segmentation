import math

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import canny, peak_local_max
from skimage.filters import sobel
from skimage.segmentation import watershed, felzenszwalb, slic, quickshift
from skimage.color import label2rgb

def show_img_with_matplotlib(img, title, pos, isBAndW, isSpectral = False):
    """Shows an image using matplotlib capabilities"""

    ax = plt.subplot(3, 2, pos)
    if isSpectral:
        plt.imshow(img, cmap = plt.cm.nipy_spectral)
    elif isBAndW:
        plt.imshow(img, cmap='gray')
    else:
    # Convert BGR image to RGB
        img_RGB = img[:, :, ::-1]
        plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

class ImagesTreatment():
    def __init__(self, image):
        self.image = image
        self.grain_images = []

    def separator(self):

        elevation_map = sobel(self.image)

        fig, ax = plt.subplots(figsize=(1, 3))
        ax.imshow(elevation_map, cmap=plt.cm.gray, interpolation='nearest')
        ax.axis('off')
        ax.set_title('elevation_map')
        return

    def basic_thresh(self):
        show_img_with_matplotlib(self.image, 'base', 1, False)

        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(image_gray, 0, 256,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        show_img_with_matplotlib(thresh, 'thresh', 2, True)

        # Contour detection
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)

        # Draw contours
        image_copy = self.image.copy()
        cv2.drawContours(image=image_copy, contours=cnts, contourIdx=-1, color=(0, 255, 0), thickness=2,
                         lineType=cv2.LINE_AA)

        show_img_with_matplotlib(image_copy, 'contours', 3, False)

    def edge_based_segmentation(self):
        show_img_with_matplotlib(self.image, 'base', 1, False)
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        edges = canny(image_gray / 255.)
        show_img_with_matplotlib(edges, 'canny', 2, True)

        filled = ndi.binary_fill_holes(edges)
        show_img_with_matplotlib(filled, 'filled', 3, True)

    def region_based_segmentation(self):
        show_img_with_matplotlib(self.image, 'base', 1, False)
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        elevation_map = sobel(image_gray)
        show_img_with_matplotlib(elevation_map, 'elevation_map', 2, True)

        markers = np.zeros_like(image_gray)
        markers[image_gray < 40] = 1
        markers[image_gray > 40] = 2
        markers[image_gray > 90] = 3

        show_img_with_matplotlib(markers, 'markers', 3, True)
        segmentation = ndi.watershed_ift(elevation_map.astype(np.uint8), markers.astype(np.int8))
        show_img_with_matplotlib(segmentation, 'segmentation', 4, True)
        segmentation = ndi.binary_fill_holes(segmentation - 1)

        labeled_coins, _ = ndi.label(segmentation)
        image_label_overlay = label2rgb(labeled_coins, image=self.image)
        show_img_with_matplotlib(image_label_overlay, 'overlay', 5, True)

    def watershed_segmentation(self):
        show_img_with_matplotlib(self.image, 'base', 1, False)

        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(image_gray, 30, 256, cv2.THRESH_BINARY)[1]

        show_img_with_matplotlib(thresh, 'thresh', 2, True)

        #Remove small objects
        distance = ndi.distance_transform_edt(thresh)
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=thresh)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=thresh)

        show_img_with_matplotlib(-distance, 'distance', 3, True)
        show_img_with_matplotlib(labels, 'labels', 4, True, isSpectral=True)


    def superpixel_segmentation(self):

        segments_fz = felzenszwalb(self.image, scale=100, sigma=0.5, min_size=50)
        segments_slic = slic(self.image, n_segments=30, compactness=10, sigma=1)
        segments_quick = quickshift(self.image, kernel_size=3, max_dist=50, ratio=0.5)

        show_img_with_matplotlib(self.image, 'base', 1, False)
        show_img_with_matplotlib(segments_fz, 'felzenszwalb', 2, True, True)
        show_img_with_matplotlib(segments_slic, 'slic', 3, True, True)
        show_img_with_matplotlib(segments_quick, 'quickshift', 4, True, True)

    def final_segmentation(self):

        show_img_with_matplotlib(self.image, 'base', 1, False)

        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(image_gray, 30, 256, cv2.THRESH_BINARY)[1]

        show_img_with_matplotlib(thresh, 'thresh', 2, True)

        distance = ndi.distance_transform_edt(thresh)
        show_img_with_matplotlib(-distance, 'distance', 3, True)

        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=thresh)
        coords = self.merge_points_too_close(coords, min_dist=9)

        coords_img = np.zeros_like(image_gray)
        for coord in coords:
            cv2.circle(coords_img, (coord[0], coord[1]), 10, (255, 0, 0), -1)
        show_img_with_matplotlib(coords_img, 'coord_after_removal', 4, True)

        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=thresh)

        for label in np.unique(labels):
        # if the label is zero label is background
            if label == 0:
                continue
            mask = np.zeros(image_gray.shape, dtype="uint8")
            mask[labels == label] = 255

            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
            cnts = imutils.grab_contours(cnts)

            for cnt in cnts:
                if cv2.contourArea(cnt) > 1500:
                    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    resultimage = cv2.bitwise_and(self.image,  mask_rgb)

                    x, y, w, h = cv2.boundingRect(cnt)
                    cropped_img = resultimage[y:y + h, x:x + w]
                    self.grain_images.append(cropped_img)
                    # cv2.drawContours(image=self.image, contours=cnts, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        show_img_with_matplotlib(labels, 'labels', 5, True, isSpectral=True)
        show_img_with_matplotlib(self.image, 'final', 6, False)

        # plt.show()
        return self.grain_images

        # show_img_with_matplotlib(self.image, 'base', 1, False)
        #
        # image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # thresh = cv2.threshold(image_gray, 30, 256, cv2.THRESH_BINARY)[1]
        # show_img_with_matplotlib(thresh, 'thresh', 2, True)
        # # noise removal
        # kernel = np.ones((3, 3), np.uint8)
        # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        # # sure background area
        # sure_bg = cv2.dilate(opening, kernel, iterations=3)
        # # Finding sure foreground area
        # show_img_with_matplotlib(sure_bg, 'background', 3, True)
        # dist_transform = ndi.distance_transform_edt(thresh)
        #
        # coords = peak_local_max(dist_transform, footprint=np.ones((3, 3)), labels=thresh)
        #
        # sure_fg = np.zeros_like(image_gray)
        #
        # coords = self.remove_point_too_close(coords, min_dist=5)
        #
        # for coord in coords:
        #
        #     cv2.circle(sure_fg, (coord[0], coord[1]), 10, (255, 0, 0), -1)
        # show_img_with_matplotlib(sure_fg, 'coord', 2, True)
        #
        # #ret, sure_fg = cv2.threshold(dist_transform, 20, 255, 0)
        # show_img_with_matplotlib(sure_fg, 'foreground', 4, True)
        # # Finding unknown region
        # sure_fg = np.uint8(sure_fg)
        # unknown = cv2.subtract(sure_bg, sure_fg)
        # # Marker labelling
        # ret, markers = cv2.connectedComponents(sure_fg)
        # # Add one to all labels so that sure background is not 0, but 1
        # markers = markers + 1
        # # Now, mark the region of unknown with zero
        # markers[unknown == 255] = 0
        # show_img_with_matplotlib(markers, 'markers', 5, True)
        # markers = cv2.watershed(self.image, markers)
        # self.image[markers == -1] = [255, 0, 0]
        #
        #
        # show_img_with_matplotlib(markers, 'final', 5, True)
        #
        # for label in np.unique(markers):
        #     # if the label is zero, we are examining the 'background'
        #     # so simply ignore it
        #     if label == 0:
        #         continue
        #     # otherwise, allocate memory for the label region and draw
        #     # it on the mask
        #     mask = np.zeros(image_gray.shape, dtype="uint8")
        #     mask[markers == label] = 255
        #     # detect contours in the mask and grab the largest one
        #     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        #                             cv2.CHAIN_APPROX_SIMPLE)
        #     cnts = imutils.grab_contours(cnts)
        #     c = max(cnts, key=cv2.contourArea)
        #     # draw a circle enclosing the object
        #     ((x, y), r) = cv2.minEnclosingCircle(c)
        #     cv2.drawContours(image=self.image, contours=cnts, contourIdx=-1, color=(0, 255, 0), thickness=2,
        #                      lineType=cv2.LINE_AA)
        #     # cv2.imshow("Output", self.image)
        #     # cv2.waitKey(0)
        #
        # show_img_with_matplotlib(self.image, 'final', 6, False)






        # # Remove small objects
        # distance = ndi.distance_transform_edt(thresh)
        # show_img_with_matplotlib(distance, 'distance', 3, True)



        # coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=thresh)
        #
        # coords_img = np.zeros_like(image_gray)
        #
        # coords = self.remove_point_too_close(coords, min_dist=5)
        #
        # for coord in coords:
        #     cv2.circle(coords_img, (coord[0], coord[1]), 10, (255, 0, 0), -1)
        # show_img_with_matplotlib(coords_img, 'coord', 2, True)
        #
        # mask = np.zeros(distance.shape, dtype=bool)
        # mask[tuple(coords.T)] = True
        # markers, _ = ndi.label(coords_img)
        # labels = cv2.watershed(self.image, markers)
        #
        # show_img_with_matplotlib(labels, 'labels', 4, True, isSpectral=True)
        # # show_img_with_matplotlib(labels, 'labels', 3, False)
        #
        # for label in np.unique(labels):
        #     # if the label is zero, we are examining the 'background'
        #     # so simply ignore it
        #     if label == 0:
        #         continue
        #     # otherwise, allocate memory for the label region and draw
        #     # it on the mask
        #     mask = np.zeros(image_gray.shape, dtype="uint8")
        #     mask[labels == label] = 255
        #     # detect contours in the mask and grab the largest one
        #     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        #                             cv2.CHAIN_APPROX_SIMPLE)
        #     cnts = imutils.grab_contours(cnts)
        #     c = max(cnts, key=cv2.contourArea)
        #     # draw a circle enclosing the object
        #     ((x, y), r) = cv2.minEnclosingCircle(c)
        #     cv2.drawContours(image=self.image, contours=cnts, contourIdx=-1, color=(0, 255, 0), thickness=2,
        #                      lineType=cv2.LINE_AA)
        #     # cv2.imshow("Output", self.image)
        #     # cv2.waitKey(0)


    def preproces_image(self, img):
        show_img_with_matplotlib(img, 'base', 1, False)

        img = self.increase_brightness(img)
        show_img_with_matplotlib(img, 'brightness_enhanced', 2, False)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = self.opening_closing(img, 5, 5)
        show_img_with_matplotlib(img, 'open_close', 2, True)
        plt.show()
        return img

    def increase_brightness(self, img, value=20):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def opening_closing(self, img, open_kernel, close_kernel):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        clos = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel2)
        return clos

    def remove_point_too_close(self, coords, min_dist=10):
        coords = coords.tolist()
        for coord in coords:
            for coord2 in coords:
                if coord[0] != coord2[0] and coord[1] != coord2[1]:
                    if math.sqrt((coord[0] - coord2[0])**2 + (coord[1] - coord2[1])**2) < min_dist:
                        coords.remove(coord2)

        return np.array(coords)

    def merge_points_too_close(self, coords, min_dist=10):
        coords = coords.tolist()

        while True:
            new_coords = []
            for coord in coords:
                close_points = []
                for coord2 in coords:
                    if coord[0] != coord2[0] and coord[1] != coord2[1]:
                        if math.sqrt((coord[0] - coord2[0])**2 + (coord[1] - coord2[1])**2) < min_dist:
                            close_points.append(coord2)
                            coords.remove(coord2)
                if len(close_points) > 0:
                    mean_x = 0
                    mean_y = 0
                    for point in close_points:
                        mean_x += point[0]
                        mean_y += point[1]
                    new_point = [int(mean_x/(len(close_points)+1)), int(mean_y/(len(close_points)+1))]
                    new_coords.append(new_point)
                    coords.remove(coord)
                else:
                    new_coords.append(coord)
            if coords == new_coords:
                break

        return np.array(new_coords)