import argparse
import numpy  as np
from glob import glob
import pandas as pd
import os

# from models.SqueezeNet import SqueezeNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify dataset with trained model.')
    parser.add_argument('dataset',
                        help='Path to dataset directory. Directory must contain a metadata.csv file and an images subdirectory containing training images.',
                        nargs='?', default='.')
    parser.add_argument('output', help='Path to output CSV file.', nargs='?', default='./classification.csv')
    args = parser.parse_args()
    print(args.dataset)
    csv_files = glob(os.path.join(args.dataset, '*.csv'))
    if len(csv_files) != 1:
        raise ValueError('The directory \"{}\" should contain exactly one csv file'.format(args.dataset))
    df = pd.read_csv(csv_files[0])
    print(df)
    # model = SqueezeNet(n_class=2, epochs=1, batch_size=1)
    # model.predict()
    # np.savetxt('classification.csv', prediction, delimiter = ',')
