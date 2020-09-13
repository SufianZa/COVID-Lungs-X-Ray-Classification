from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from utils.customImageGenerator import CustomImageGenerator

def show_learning_curves(history, model_name):
    fig, ax = plt.subplots(2, 1, figsize=(7, 12), facecolor='#F0F0F0')
    ax[0].plot(history['accuracy'])
    ax[0].plot(history['val_accuracy'])
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
    plt.savefig('{}_curves.png', model_name)


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
    plt.savefig('{}_conf_matrix.png'.format(model_name))



class BaseModel:
    def __init__(self, input_size=(384, 384, 1), n_class=3, batch_size=16, epochs=60, model_name='undefined'):
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.CLASS_TARGETS = ['No Finding', 'Covid', 'other'] if n_class == 3 else ['No Finding', 'Covid',
                                                                                    'Enlarged Cardiomediastinum',
                                                                                    'Cardiomegaly', 'Lung Opacity',
                                                                                    'Lung Lesion', 'Edema',
                                                                                    'Consolidation', 'Pneumonia',
                                                                                    'Atelectasis',
                                                                                    'Pneumothorax', 'Pleural Effusion',
                                                                                    'Pleural Other', 'Fracture',
                                                                                    'Support Devices']
        self.input_size = input_size
        self.weight_file = 'weight{}.hdf5'.format(model_name)
        self.init_network()

    def train(self, dir, pkl_file):
        # load files, label, features
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)  # 0: files, 1: features, 2: labels
        if len(self.CLASS_TARGETS) == 3:
            df[3] = [[l[0], l[1], l[2:].sum()] for l in df[2].tolist()]  # 3: labels summarized into 3 e. g. [0 , 0 , 1]
        else:
            df[3] = df[2]
        df = df[:100]
        # split data into 80% train, 15% validation and 5% unseen test data
        n_samples = len(df)
        train_x, val_x, train_y, val_y = train_test_split(df[0].tolist(), df[3].tolist(), test_size=0.2,
                                                          random_state=2134, stratify=df[3].tolist())

        print('Number of samples: {} --- {}% train data | {}% validation data'.format(n_samples, (
                len(train_x) / n_samples) * 100, (len(val_x) / n_samples) * 100))

        # create generators

        train_generator = CustomImageGenerator(train_x, train_y, directory=dir, batch_size=self.batch_size,
                                               input_size=self.input_size)
        valid_generator = CustomImageGenerator(val_x, val_y, directory=dir, batch_size=self.batch_size,
                                               input_size=self.input_size)

        check_point = ModelCheckpoint('best_' + self.weight_file, monitor='val_loss', mode='min', save_best_only=True,
                                      verbose=1)

        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=5,
                                   verbose=0, mode='auto')

        # fit model
        self.history = self.model.fit(train_generator, validation_data=valid_generator, epochs=self.epochs,
                                      steps_per_epoch=train_generator.n // self.batch_size,
                                      validation_steps=valid_generator.n // self.batch_size,
                                      callbacks=[check_point])

        # save weight and test data
        self.model.save_weights(self.weight_file)
        with open('../history_{}.pkl'.format(self.model_name), "wb") as f:
            pickle.dump({'history': self.history.history}, f)

    def evaluate_model(self, dir, pkl_file):
        self.model.load_weights('best_' + self.weight_file)
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)  # 0: files, 1: features, 2: labels
        if len(self.CLASS_TARGETS) == 3:
            df[3] = [[l[0], l[1], l[2:].sum()] for l in df[2].tolist()]  # 3: labels summarized into 3 e. g. [0 , 0 , 1]
        else:
            df[3] = df[2]
        df = df[:200]
        unseen_gen = CustomImageGenerator(df[0].tolist(), df[3].tolist(), directory=dir, batch_size=1, shuffle=False,
                                          input_size=self.input_size)

        predictions = self.model.predict(unseen_gen, workers=0, use_multiprocessing=False, verbose=0)

        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(unseen_gen.labels, axis=1)

        show_confusion_Matrix(y_pred, y_true, self.CLASS_TARGETS, self.model_name)

        try:
            with open('../history_{}.pkl'.format(self.model_name), "rb") as f:
                data = pickle.load(f)
            show_learning_curves(data['history'], self.model_name)
        except:
            print('No history of learning_curves found')


        plt.show()

    def init_network(self):
        raise NotImplementedError

    def evaluate_model2(self, dir):
        self.model.load_weights('best_' + self.weight_file)
        with open("../unseen_data.pkl", "rb") as f:
            data = pickle.load(f)

        unseen_gen = CustomImageGenerator(data['x'], data['y'], directory=dir, batch_size=1, shuffle=False,
                                          input_size=self.input_size)

        for x in range(unseen_gen.n):
            item = unseen_gen.__getitem__(x)
            if item[0]:
                image = np.squeeze(item[0])
                print(image.shape)
                label = np.squeeze(item[1])
                # predictions = self.model.predict(image, label, workers=0, use_multiprocessing=False, verbose=0)
                plt.imshow(image, cmap='gray')
                plt.show()
            # y_pred = np.argmax(predictions, axis=1)
            # y_true = np.argmax(unseen_gen.labels, axis=1)

        show_confusion_Matrix(y_pred, y_true, self.CLASS_TARGETS, self.model_name)
        # plt.savefig('conf_matrix.png')

        show_learning_curves(data['history'], self.model_name)
