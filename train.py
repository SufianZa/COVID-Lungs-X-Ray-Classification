import argparse
from models.SqueezeNet import SqueezeNet
from glob import glob
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model with given dataset.')
    parser.add_argument('dataset', help='Path to dataset directory. Directory must contain a metadata.csv file and an images subdirectory containing training images.', nargs='?', default='.')
    args = parser.parse_args()

    args = parser.parse_args()

    csv_files = glob(os.path.join(args.dataset, '*.csv'))
    if len(csv_files) != 1:
        raise ValueError('The directory \"{}\" should contain exactly one csv file'.format(args.dataset))

    model = SqueezeNet(n_class=3, epochs=1, batch_size=1)
    # You need to run preprocessing in order to use pkl file applying offline augmentation data.
    # otherwise the data is read from the disk directly
    model.train(dir=args.dataset, pkl_file='train_paths_features_labels.pkl')

