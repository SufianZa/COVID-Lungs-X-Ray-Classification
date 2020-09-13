import argparse
from keras.optimizers import Adam
from models.SqueezeNet import SqueezeNet
from models.DenseNet201 import DenseNet201
from models.ResNet152 import ResNet152

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model with given dataset.')
    parser.add_argument('dataset', help='Path to dataset directory. Directory must contain a metadata.csv file and an images subdirectory containing training images.', nargs='?', default='.')
    args = parser.parse_args()

    model = DenseNet201(n_class=15, epochs=1, batch_size=1)
    model.train(dir='train_data', pkl_file='train_paths_features_labels.pkl')
    model.evaluate_model(dir='train_data', pkl_file='test_paths_features_labels.pkl')

    model = SqueezeNet(n_class=15, epochs=1, batch_size=1)
    model.train(dir='train_data', pkl_file='train_paths_features_labels.pkl')
    model.evaluate_model(dir='train_data', pkl_file='test_paths_features_labels.pkl')

    model = ResNet152(n_class=15, epochs=1, batch_size=1)
    model.train(dir='train_data', pkl_file='train_paths_features_labels.pkl')
    model.evaluate_model(dir='train_data', pkl_file='test_paths_features_labels.pkl')
