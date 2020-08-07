import os
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
from utils.augmentation import save_augmentation
from skimage import exposure
import matplotlib.pyplot as plt
import pickle


class DataPreparation:
    def __init__(self, csv_path, src_dir, dst_dir):
        self.IMAGE_SIZE = (384, 384)
        self.CLASS_TARGETS = ['No Finding', ' Other', 'COVID-19']
        self.CLASS_NUM = len(self.CLASS_TARGETS)
        self.images_list = glob(os.path.join(src_dir, '*'), recursive=True)
        self.df = pd.read_csv(csv_path)
        self.dst_dir = dst_dir
        if not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir)

    def load(self):
        num_covid_pre_aug = 0
        num_other_pre_aug = 0
        num_no_finding_pre_aug = 0
        data = {'paths': [], 'features': [], 'labels': []}
        num_aug_images = 0
        for image_path in self.images_list:
            image_name = os.path.basename(image_path)
            # extract features from csv
            csv_row = self.df.loc[self.df.File == image_name]
            features = np.array([*csv_row['Age'].values, csv_row['Sex'].values])

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
                num_covid_pre_aug +=1
            if other_disease:
                label[1] = 1
                num_other_pre_aug  += 1
            if no_finding:
                num_no_finding_pre_aug +=1
                label[0] = 1

            # throw error if label not vaild
            if np.count_nonzero(label) != 1:
                raise ValueError('CSV Data Error: Found no corresponding label in row {}'.format(csv_row))

            if covid:
                save_augmentation(image, image_name, label, features, data, self.dst_dir)

            if no_finding and num_aug_images < 600:
                save_augmentation(image, image_name, label, features, data, self.dst_dir, aug_num=1)
                num_aug_images += 1

            # add to data object
            data['paths'].append(image_name)
            data['features'].append(features)
            data['labels'].append(label)
            result = Image.fromarray((image * 255).astype(np.uint8))
            result.save(os.path.join(self.dst_dir, image_name))
        with open('paths_features_labels.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        f, axs = plt.subplots(1,2,figsize=(12,4))
        axs[0].bar('Covid', num_covid_pre_aug)
        axs[0].bar('Other disease', num_other_pre_aug)
        axs[0].bar('No Finding', num_no_finding_pre_aug)
        axs[0].set_title('Labels before Augmentation')
        axs[0].set_xlabel('Label')
        axs[0].set_ylabel('Count')
        _, counts = np.unique(data['labels'], axis=0, return_counts=True)
        axs[1].bar('Covid', counts[0])
        axs[1].bar('Other disease', counts[1])
        axs[1].bar('No Finding', counts[2])
        axs[1].set_title('Labels after Augmentation')
        axs[1].set_xlabel('Label')
        axs[1].set_ylabel('Count')
        plt.tight_layout()
        plt.savefig('balancing.png')

if __name__ == '__main__':
    DataPreparation(csv_path='metadata.csv', src_dir='images', dst_dir='train_data').load()
