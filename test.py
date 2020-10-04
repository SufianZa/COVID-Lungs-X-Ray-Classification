import argparse
from glob import glob
import pandas as pd
import os
from models.SqueezeNet import SqueezeNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify dataset with trained model.')
    parser.add_argument('dataset',
                        help='Path to dataset directory. Image files could be in \'jpg\' or \'png\' format',
                        nargs='?', default='.')

    parser.add_argument('output', help='Path to output CSV file.', nargs='?', default='results.csv')
    args = parser.parse_args()
    files_list = glob(os.path.join(args.dataset, '*.jpg')) + glob(os.path.join(args.dataset, '*.jpeg')) + glob(os.path.join(args.dataset, '*.png'))

    # for n_class=2 the predictions are either ['Covid', 'Non-Covid']
    # for n_class=3 the predictions consists of ['No Finding', 'Covid', 'other']
    model = SqueezeNet(n_class=3, epochs=1, batch_size=1)
    predictions = model.predict(files_list=files_list)
    pd.DataFrame(data={'files': files_list, 'predictions': predictions}).to_csv(args.output, index=False, header=True)
