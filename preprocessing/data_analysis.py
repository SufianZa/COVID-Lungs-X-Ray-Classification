import os
import argparse
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


def clean_data(df, print_info=False):
    df.dropna(inplace=True)
    assert pd.notnull(df).all().all()
    if print_info:
        print(df.info())
        print(df.isnull().sum())
        # print csv features
        print(df.columns)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze given dataset producing console output and figures describing the data.')
    parser.add_argument('dataset',
                        help='Path to dataset directory. Directory must contain a metadata.csv file and an images subdirectory containing training images.',
                        nargs='?', default='.')
    parser.add_argument('output', help='Path to output directory.', nargs='?', default='./results')
    args = parser.parse_args()
    # Read the image list and csv as Dataframe
    images_path = os.path.join(args.dataset)
    image_files = glob(os.path.join(images_path, '*.jpg')) + glob(os.path.join(images_path, '*.jpeg')) + glob(os.path.join(images_path, '*.png'))

    df = pd.read_csv(os.path.join(args.dataset, 'metadata.csv'))

    clean_data(df, print_info=True)

    colored_imgs = []
    gray_imgs = []
    for im in image_files:
        image = np.array(Image.open(im))
        if image.ndim > 2:
            colored_imgs.append((image.shape[0], image.shape[1], image.shape[2]))
        else:
            gray_imgs.append((image.shape[0], image.shape[1], None))

    all_imgs = np.array(colored_imgs + gray_imgs)
    colored_imgs = np.array(colored_imgs)
    gray_imgs = np.array(gray_imgs)

    print('RGB images {}'.format(len(colored_imgs)))
    print('Grayscale images {}'.format(len(gray_imgs)))

    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(121)
    ax1.scatter(colored_imgs[:, 0], colored_imgs[:, 1], c='r', label='RGB Images')
    ax1.scatter(gray_imgs[:, 0], gray_imgs[:, 1], c='gray', label='Grayscale Images')
    ax1.grid()
    ax1.set_xlabel('Width', fontsize=15)
    ax1.set_ylabel('Height', fontsize=15)
    ax1.set_title('Image Dimensions (in Pixels)', fontsize=12)
    ax1.legend()
    ax1 = fig.add_subplot(122)

    ax1.hist(all_imgs[:, :1], bins=10)
    ax1.grid()
    ax1.set_xlabel('Image Size', fontsize=12)
    ax1.set_ylabel('Num of Images (log)', fontsize=12)
    ax1.xaxis.set_major_formatter(FuncFormatter((lambda x, b: '({},{})'.format(int(x), int(x)))))
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.xaxis.set_tick_params(labelsize=9)
    ax1.set_title('Image Size Distribution', fontsize=12)
    plt.savefig(args.output + '/image_sizes.png')

    plt.figure(figsize=(13, 5))
    plt.subplot(121)
    male_ratio = df[(df["Covid"] == 1) & (df["Sex"] == 'Male')]
    female_ratio = df[(df["Covid"] == 1) & (df["Sex"] == 'Female')]
    plt.bar('Male', male_ratio.shape[0])
    plt.bar('Female', female_ratio.shape[0])
    plt.ylabel('COVID-19 Cases', fontsize=12)

    covid = df[(df["Covid"] == 1)]
    plt.subplot(122)
    covid['Age'].hist(bins=20)
    plt.xlabel('Age', fontsize=12)
    plt.savefig(args.output + '/features_analysis.png')

    print('Average age of Covid patients: {}'.format(covid["Age"].mean()))

    print('Train images {}'.format(len(df)))
