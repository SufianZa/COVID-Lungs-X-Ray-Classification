import os
import argparse
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image

from tqdm import tqdm
from preprocessing.augmentation import save_augmentation, save_image
from skimage import exposure
import matplotlib.pyplot as plt
import pickle


class Preprocessing:
    def __init__(self, csv_path, src_dir, dst_dir, figure_dst_dir):

        self.IMAGE_SIZE = (320, 320)
        self.CLASS_TARGETS = ['No Finding', 'Covid', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
        self.label_counter = np.zeros(len(self.CLASS_TARGETS))
        self.CLASS_NUM = len(self.CLASS_TARGETS)
        self.images_list = glob(os.path.join(src_dir, '*'), recursive=True)
        self.df = pd.read_csv(csv_path)
        self.dst_dir = dst_dir
        self.figure_dst_dir = figure_dst_dir
        if dst_dir and not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir)

    def load(self):
        data = []
        test_data = []
        no_finding_aug = 0
        for image_path in tqdm(self.images_list):
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
            if 100 > self.label_counter[np.argmax(label)] > 80:
                save_image(image, image_name, self.dst_dir)
                test_data.append((image_name, features, label))
                self.label_counter = self.label_counter + label
                continue
            elif self.label_counter[np.argmax(label)] > 100:  # discard images above 100
                continue
            self.label_counter = self.label_counter + label

            # throw error if label not valid
            if np.count_nonzero(label) != 1:
                raise ValueError('CSV Data Error: Found no corresponding label in row {}'.format(csv_row))

            save_augmentation(image, image_name, label, features, data, self.dst_dir, aug_num=3)

            data.append((image_name, features, label))
            save_image(image, image_name, self.dst_dir)

        with open('../train_paths_features_labels.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        with open('../test_paths_features_labels.pkl', 'wb') as f:
            pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
        self.plot_bar(data)

    def get_files_and_labels(self):
        data = []
        for image_path in tqdm(self.images_list):
            image_name = os.path.basename(image_path)
            # extract features from csv
            if (self.df.File == image_name).any():
                csv_row = self.df.loc[self.df.File == image_name]
                features = np.array([*csv_row['Age'].values, csv_row['Sex'].values])
                # extract label
                label = np.array(np.array(csv_row.iloc[0][self.CLASS_TARGETS]) > 0).astype(int)

                # throw error if label not valid
                if np.count_nonzero(label) != 1:
                    raise ValueError('CSV Data Error: Found no corresponding label in row {}'.format(csv_row))

                data.append((image_name, features, label))
        return np.array(data)

    def plot_bar(self, data):
        df = pd.DataFrame(data)
        f, axs = plt.subplots(2, 2, figsize=(12, 12))
        self.CLASS_TARGETS[2] = 'Enlarged Cardi.'
        axs[0][0].bar(self.CLASS_TARGETS, self.label_counter)
        axs[0][0].set_title('Labels before Augmentation')
        axs[0][0].set_xlabel('Label')
        axs[0][0].set_xticklabels(self.CLASS_TARGETS, rotation=90, fontsize=9)
        axs[0][0].set_ylabel('Count')
        _, counts = np.unique(df[2].tolist(), axis=0, return_counts=True)
        counts = counts[::-1]
        axs[0][1].bar(self.CLASS_TARGETS, counts)
        axs[0][1].set_title('Labels after Augmentation')
        axs[0][1].set_xlabel('Label')
        axs[0][1].set_ylabel('Count')

        axs[1][0].bar(self.CLASS_TARGETS[:2] + ['Other'], [*self.label_counter[:2], self.label_counter[2:].sum()])
        axs[1][0].set_title('Labels before Augmentation Summarized 3 classes')
        axs[1][0].set_xlabel('Label')
        axs[1][0].set_ylabel('Count')

        axs[1][1].bar(self.CLASS_TARGETS[:2] + ['Other'], [*counts[:2], counts[2:].sum()])
        axs[1][1].set_title('Labels after Augmentation Summarized 3 classes')
        axs[1][1].set_xlabel('Label')
        axs[1][1].set_ylabel('Count')

        plt.tight_layout()
        axs[0][1].set_xticklabels(self.CLASS_TARGETS, rotation=90, fontsize=9)
        plt.savefig(self.figure_dst_dir + '/balancing.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess and augment images of given dataset. Additionally figures will be created.')
    parser.add_argument('dataset',
                        help='Path to dataset directory. Directory must contain a metadata.csv file and an images subdirectory containing training images.',
                        nargs='?', default='.')
    parser.add_argument('output', help='Path to image output directory.', nargs='?', default='./train_data')
    parser.add_argument('figure_output', help='Path to figure output directory.', nargs='?', default='./results')
    args = parser.parse_args()
    Preprocessing(csv_path=(args.dataset + '/metadata.csv'), src_dir=(args.dataset + '/images'), dst_dir=args.output, figure_dst_dir=args.figure_output).load()
