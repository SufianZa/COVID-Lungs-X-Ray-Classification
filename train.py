import argparse
from model import SqueezeNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model with given dataset.')
    parser.add_argument('dataset', help='Path to dataset directory. Directory must contain a metadata.csv file and an images subdirectory containing training images.', nargs='?', default='.')
    args = parser.parse_args()