import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify dataset with trained model.')
    parser.add_argument('dataset', help='Path to dataset directory. Directory must contain a metadata.csv file and an images subdirectory containing training images.', nargs='?', default='.')
    parser.add_argument('output', help='Path to output CSV file.', nargs='?', default='./classification.csv')
    args = parser.parse_args()