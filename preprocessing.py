import os
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image

from model import SqueezeNet
from utils.augmentation import save_augmentation
from skimage import exposure
import matplotlib.pyplot as plt
import pickle

class Preprocessing:
    def __init__(self, csv_path, src_dir, dst_dir):
        self.IMAGE_SIZE = (384, 384)
        self.CLASS_TARGETS = ['No Finding', 'Covid', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                           'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                           'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
        self.label_counter = np.zeros(len(self.CLASS_TARGETS))
        self.CLASS_NUM = len(self.CLASS_TARGETS)
        self.images_list = glob(os.path.join(src_dir, '*'), recursive=True)
        self.df = pd.read_csv(csv_path)
        self.dst_dir = dst_dir
        if not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir)

    def load(self):
        data = {'files': [], 'features': [], 'labels': []}
        no_finding_aug = 0
        for image_path in self.images_list:
            image_name = os.path.basename(image_path)
            # extract features from csv
            csv_row = self.df.loc[self.df.File == image_name]
            features = np.array([*csv_row['Age'].values, csv_row['Sex'].values])

            # load image
            image = np.array(Image.open(image_path).resize(self.IMAGE_SIZE))

            # convert RGB to grayscale
            if image.ndim > 2:
                image = np.mean(image, axis=2)

            # Histogram equalization
            image = exposure.equalize_hist(image)

            # extract label
            label = np.array(np.array(csv_row.iloc[0][self.CLASS_TARGETS]) > 0).astype(int)

            self.label_counter = self.label_counter + label

            # throw error if label not valid
            if np.count_nonzero(label) != 1:
                raise ValueError('CSV Data Error: Found no corresponding label in row {}'.format(csv_row))

            if label[0] and no_finding_aug < 600:
                save_augmentation(image, image_name, label, features, data, self.dst_dir, aug_num=1)
                no_finding_aug += 1

            if label[1]:
                save_augmentation(image, image_name, label, features, data, self.dst_dir)

            # add to data object
            data['files'].append(image_name)
            data['features'].append(features)
            data['labels'].append(label)
            result = Image.fromarray((image * 255).astype(np.uint8))
            result.save(os.path.join(self.dst_dir, image_name))
        with open('paths_features_labels.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        f, axs = plt.subplots(2,2,figsize=(12,12))
        self.CLASS_TARGETS[2] = 'Enlarged Cardi.'
        axs[0][0].bar(self.CLASS_TARGETS, self.label_counter)
        axs[0][0].set_title('Labels before Augmentation')
        axs[0][0].set_xlabel('Label')
        axs[0][0].set_xticklabels(self.CLASS_TARGETS, rotation=90, fontsize=9)
        axs[0][0].set_ylabel('Count')
        _, counts = np.unique(data['labels'], axis=0, return_counts=True)

        counts = counts[::-1]
        axs[0][1].bar(self.CLASS_TARGETS, counts)
        axs[0][1].set_title('Labels after Augmentation')
        axs[0][1].set_xlabel('Label')
        axs[0][1].set_ylabel('Count')

        axs[1][0].bar(self.CLASS_TARGETS[:2] +[ 'Other'], [*self.label_counter[:2], self.label_counter[2:].sum()])
        axs[1][0].set_title('Labels before Augmentation Summarized 3 classes')
        axs[1][0].set_xlabel('Label')
        axs[1][0].set_ylabel('Count')

        axs[1][1].bar(self.CLASS_TARGETS[:2] +['Other'], [*counts[:2] , counts[2:].sum()])
        axs[1][1].set_title('Labels after Augmentation Summarized 3 classes')
        axs[1][1].set_xlabel('Label')
        axs[1][1].set_ylabel('Count')

        plt.tight_layout()
        axs[0][1].set_xticklabels(self.CLASS_TARGETS, rotation=90, fontsize=9)
        plt.savefig('balancing.png')

if __name__ == '__main__':
    Preprocessing(csv_path='metadata.csv', src_dir='images', dst_dir='train_data').load()
