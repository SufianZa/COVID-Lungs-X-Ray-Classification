import os
import random
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from skimage import exposure


class DataPreparation:
    def __init__(self, csv_path, images_path):
        self.IMAGE_SIZE = (384,384)
        self.CLASS_TARGETS = ['No Finding', ' Other', 'COVID-19']
        self.CLASS_NUM = len(self.CLASS_TARGETS)
        self.images_list = glob(images_path, recursive=True)
        self.df = self.clean_data(pd.read_csv(csv_path))

    def clean_data(self, df):
        df.dropna(inplace=True)
        assert pd.notnull(df).all().all()
        return df

    def load(self):
        data = {'images': [], 'features': [], 'labels': []}
        for image_path in self.images_list:
            image_name = os.path.basename(image_path)
            # extract features from csv
            csv_row = self.df.loc[self.df.File == image_name]
            if csv_row.empty : continue

            features = np.array([*csv_row['Age'].values, 0 if csv_row['Sex'].values == 'Female' else 1])
            image = np.array(Image.open(image_path).resize(self.IMAGE_SIZE))

            # convert RGB to grayscale
            if image.ndim > 2: image = np.mean(image, axis=2)

            # Histogram equlization
            image = exposure.equalize_hist(image)

            # extract label
            other_disease = np.any(
                np.array(csv_row.iloc[0][['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                                          'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                                          'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                                          'Support Devices']]) > 0)
            covid = csv_row.iloc[0]['Covid'] > 0
            no_finding = csv_row.iloc[0]['No Finding'] > 0

            label = np.zeros(self.CLASS_NUM)
            if covid:
                label[2] = 1
            if other_disease:
                label[1] = 1
            if no_finding:
                label[0] = 1

            # throw error if label not vaild
            if np.count_nonzero(label) != 1:
                raise ValueError('CSV Data Error: Found no corresponding label in row {}'.format(csv_row))

            # add data to batch
            data['images'].append(image)
            data['features'].append(features)
            data['labels'].append(label)

        return np.array(data['images']), np.array(data['features']), np.array(data['labels'])


if __name__ == '__main__':
    X, f, y = DataPreparation('metadata.csv', 'images/*').load()