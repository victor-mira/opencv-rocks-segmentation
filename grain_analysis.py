import numpy as np
from pandas import DataFrame

class GrainAnalyser():
    def __init__(self, grains_images):
        self.grains_images = grains_images

    def compute_df(self):
        averages_colors = []
        for img in self.grains_images:

            average_color_row = np.average(img, axis=0)
            average_color = np.average(average_color_row, axis=0)
            averages_colors.append(average_color)

        df = DataFrame(averages_colors, columns=['Moyenne de B', 'Moyenne de G', 'Moyenne de R'])
        return df


