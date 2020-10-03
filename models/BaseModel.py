from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from utils.customImageGenerator import CustomImageGenerator
from preprocessing.preprocess import Preprocessing
from glob import glob


def show_learning_curves(history, model_name):
    fig, ax = plt.subplots(2, 1, figsize=(7, 12), facecolor='#F0F0F0')
    ax[0].plot(history['acc'])
    ax[0].plot(history['val_acc'])
    ax[0].set_title('model accuracy')
    ax[0].set_ylabel('accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['train', 'valid.'])

    ax[1].plot(history['loss'])
    ax[1].plot(history['val_loss'])
    ax[1].set_title('model loss')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['train', 'valid.'])
    fig.suptitle('Learning curves of {} COVID Classifier'.format(model_name), y=1)
    plt.tight_layout()
    plt.savefig('results/{}_curves.png'.format(model_name))


def show_confusion_Matrix(y_pred, y_true, targets, model_name):
    print(classification_report(y_true, y_pred, target_names=targets))

    # Create confusion matrix
    confusionMatrix = (confusion_matrix(
        y_true=y_true,  # ground truth
        y_pred=y_pred, normalize='true'))

    confusionMatrix = confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(confusionMatrix, annot=True, annot_kws={'size': 15},
                cmap=plt.cm.Blues)
    # Add labels to the plot
    tick_marks = np.arange(len(targets))
    tick_marks2 = tick_marks + 0.5
    tick_marks = tick_marks + 0.5
    plt.xticks(tick_marks, targets, rotation=0)
    plt.yticks(tick_marks2, targets, rotation=0)
    plt.xlabel('Predicted label', labelpad=15)
    plt.ylabel('True label')
    plt.title('Confusion Matrix of {} COVID Classifier'.format(model_name))
    plt.savefig('results/{}_conf_matrix.png'.format(model_name))


class BaseModel:
    def __init__(self, input_size=(320, 320, 1), n_class=3, batch_size=16, epochs=60, model_name='undefined'):
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        if n_class == 3:
            self.CLASS_TARGETS = ['No Finding', 'Covid', 'other']
            self.weight_file = 'weights/best_weight_{}_3_classes.hdf5'.format(model_name)
        elif n_class == 2:
            self.CLASS_TARGETS = ['Covid', 'Non-Covid']
            self.weight_file = 'weights/best_weight_{}_2_classes.hdf5'.format(model_name)
        else:
            self.CLASS_TARGETS = ['No Finding', 'Covid', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                                  'Lung Lesion', 'Edema',
                                  'Consolidation', 'Pneumonia',
                                  'Atelectasis',
                                  'Pneumothorax', 'Pleural Effusion',
                                  'Pleural Other', 'Fracture',
                                  'Support Devices']
            self.weight_file = 'weights/best_weight_{}_15_classes.hdf5'.format(model_name)
        self.input_size = input_size
        self.init_network()

    def train(self, dir, pkl_file):
        try:
            # load files, label, features
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        except:
            print('No Augmentation file \'{}\' found'.format(pkl_file))
            csv_file = glob(os.path.join(dir, '*.csv'))[0]
            p = Preprocessing(src_dir=dir, csv_path=csv_file)
            data = p.get_files_and_labels()
        df = pd.DataFrame(data)  # 0: files, 1: features, 2: labels
        df = self.get_data(df[0].tolist(), df[2].tolist())  # 0: files, 1: all_labels, 2: labels_summerized
        df = df[:20]

        # split data into 80% train and 20% validation
        n_samples = len(df)
        train_x, val_x, train_y, val_y = train_test_split(df[0].tolist(), df[2].tolist(), test_size=0.2,
                                                          random_state=2134, stratify=df[2].tolist())

        print('Number of samples: {} --- {}% train data | {}% validation data'.format(n_samples, (
                len(train_x) / n_samples) * 100, (len(val_x) / n_samples) * 100))

        # create generators
        train_generator = CustomImageGenerator(files=train_x, labels=train_y, directory=dir, batch_size=self.batch_size,
                                               input_size=self.input_size)
        valid_generator = CustomImageGenerator(files=val_x, labels=val_y, directory=dir, batch_size=self.batch_size,
                                               input_size=self.input_size)

        # save the best weights
        check_point = ModelCheckpoint(self.weight_file, monitor='val_loss', mode='min', save_best_only=True,
                                      verbose=1)
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=5,
                                   verbose=0, mode='auto')
        # fit model
        self.history = self.model.fit(train_generator, validation_data=valid_generator, epochs=self.epochs,
                                      steps_per_epoch=train_generator.n // self.batch_size,
                                      validation_steps=valid_generator.n // self.batch_size,
                                      callbacks=[check_point, early_stop])
        show_learning_curves(self.history.history, self.model_name)

    def evaluate_model(self, dir, pkl_file):
        self.load_weights()
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)  # 0: files, 1: features, 2: labels
        df = self.get_data(df[0], df[2])  # 0: files, 1: all_labels, 2: labels_summerized

        unseen_gen = CustomImageGenerator(files=df[0].tolist(), labels=df[2].tolist(), directory=dir, batch_size=1, shuffle=False,
                                          input_size=self.input_size)

        predictions = self.model.predict(unseen_gen, workers=0, use_multiprocessing=False, verbose=0)

        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(unseen_gen.labels, axis=1)
        show_confusion_Matrix(y_pred, y_true, self.CLASS_TARGETS, self.model_name)
        plt.show()

    def init_network(self):
        raise NotImplementedError

    def predict(self, files_list):
        self.load_weights()
        pred_gen = CustomImageGenerator(directory='', files=files_list, labels=None, batch_size=1, shuffle=False,
                                        input_size=self.input_size)

        predictions = self.model.predict(pred_gen, workers=0, use_multiprocessing=False, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        classes = np.array(self.CLASS_TARGETS)
        return classes[y_pred]

    def get_data(self, files_list, labels):
        df = pd.DataFrame({0: files_list, 1: labels})  # 0: files,  1: all_labels
        if len(self.CLASS_TARGETS) == 3:
            df[2] = [[l[0], l[1], l[2:].sum()] for l in df[1].tolist()]  # 2: labels summarized into 3 e. g. [0 , 0 , 1]
        elif len(self.CLASS_TARGETS) == 2:
            df[2] = [[l[1], l[2:].sum() + l[0]] for l in df[1].tolist()]  # 2: labels summarized into 2 [0 , 1]
        else:
            df[2] = df[1]  # 2: labels are 15 classes
        return df

    def load_weights(self):
        try:
            self.model.load_weights(self.weight_file)
        except:
            print('Weight file not Found: Need to train the model first.')
            os._exit(0)
