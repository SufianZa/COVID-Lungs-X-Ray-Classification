from tensorflow.keras.layers import Conv2D, MaxPool2D, concatenate, \
    GlobalAveragePooling2D, Dropout, Input, Activation
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
from utils.customImageGenerator import CustomImageGenerator
import pickle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def show_learning_curves(history):
    plt.subplots(figsize=(15, 20), facecolor='#F0F0F0')
    plt.tight_layout()
    ax = plt.subplot(211)
    ax.plot(history['accuracy'])
    ax.plot(history['val_accuracy'])
    ax.set_title('model accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])

    ax = plt.subplot(212)
    ax.plot(history['loss'])
    ax.plot(history['val_loss'])
    ax.set_title('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])


def show_confusion_Matrix(y_pred, y_true, targets):
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
    plt.title('Confusion Matrix of SqueezeNet COVID Classifier')


class SqueezeNet:
    def fire(self, x, squeeze_size):
        return self.expand(self.squeeze(x, squeeze_size), squeeze_size*4)

    def squeeze(self, y, squeeze_size):
        return Conv2D(filters=squeeze_size, kernel_size=1, activation='relu', padding='same')(y)

    def expand(self, x, expand_size):
        left = Conv2D(filters=expand_size, kernel_size=1, activation='relu', padding='same')(x)
        right = Conv2D(filters=expand_size, kernel_size=3, activation='relu', padding='same')(x)
        return concatenate([left, right], axis=3)

    def __init__(self, input_size=384, n_class=3):
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
        self.weight_file = 'weight.hdf5'
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

        # split data into 80% train, 15% validation and 5% unseen test data
        n_samples = len(df)
        train_x, test_x, train_y, test_y = train_test_split(df[0].tolist(), df[3].tolist(), test_size=0.2,
                                                            random_state=5855, stratify=df[3].tolist())
        val_x, unseen_x, val_y, unseen_y = train_test_split(test_x, test_y, test_size=0.25, random_state=4255,
                                                            stratify=test_y)
        print('Number of samples: {} --- {}% train data | {}% validation data | {}% test data'.format(n_samples, (
                len(train_x) / n_samples) * 100, (len(val_x) / n_samples) * 100, (len(unseen_x) / n_samples) * 100))

        # create generators
        train_generator = CustomImageGenerator(train_x, train_y, directory=dir, batch_size=16, img_dim=self.input_size)
        valid_generator = CustomImageGenerator(val_x, val_y, directory=dir, batch_size=16, img_dim=self.input_size)

        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=8,
                                   verbose=0, mode='auto')
        # fit model
        self.history = self.model.fit(train_generator, validation_data=valid_generator, epochs=70,
                                      steps_per_epoch=train_generator.n // train_generator.batch_size,
                                      validation_steps=valid_generator.n // valid_generator.batch_size,
                                      callbacks=[early_stop])

        # save waight and test data
        self.model.save_weights(self.weight_file)
        with open(pkl_file, "wb") as f:
            pickle.dump({'x': unseen_x, 'y': unseen_y, 'history': self.history.history}, f)

    def evaluate_model(self, dir):
        self.model.load_weights(self.weight_file)
        with open("unseen_data.pkl", "rb") as f:
            data = pickle.load(f)
        # data['y'] = [data['y'][idx] for idx, f in enumerate(data['x']) if os.path.isfile(os.path.join(dir, f)) and os.access(os.path.join(dir, f), os.R_OK)]
        # data['x'] = [f for idx, f in enumerate(data['x']) if os.path.isfile(os.path.join(dir, f)) and os.access(os.path.join(dir, f), os.R_OK)]
        unseen_gen = CustomImageGenerator(data['x'], data['y'], directory=dir, batch_size=1, shuffle=False)
        predictions = self.model.predict(unseen_gen, workers=0, use_multiprocessing=False, verbose=0)

        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(unseen_gen.labels, axis=1)

        show_confusion_Matrix(y_pred, y_true, self.CLASS_TARGETS)
        plt.savefig('conf_matrix.png')

        show_learning_curves(data['history'])
        plt.savefig('learning_curve.png')

        plt.show()

    def init_network(self):
        # input
        input_layer = Input((self.input_size, self.input_size, 1))

        # conv 1
        x = Conv2D(kernel_size=7, filters=96, padding='same', activation='relu', strides=2)(input_layer)

        # maxpool
        x = MaxPool2D(pool_size=3, strides=2)(x)

        # fire 2
        x = self.fire(x, 64)

        # fire 3
        x = self.fire(x, 64)

        # maxpool
        x = MaxPool2D(pool_size=3, strides=2)(x)

        # fire 4
        x = self.fire(x, 80)

        # fire 5
        x = self.fire(x, 80)

        # maxpool
        x = MaxPool2D(pool_size=3, strides=2)(x)

        # fire 6
        x = self.fire(x, 96)

        # fire 7
        x = self.fire(x, 96)

        # fire 8
        x = self.fire(x, 112)

        # fire 9
        x = self.fire(x, 112)

        x = Dropout(0.5)(x)

        # conv 10
        x = Conv2D(kernel_size=1, filters=len(self.CLASS_TARGETS))(x)

        # global avgpool
        x = GlobalAveragePooling2D()(x)

        # softmax
        x = Activation('softmax')(x)

        # create Model
        self.model = Model(input_layer, x)

        # compile Model
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(lr=0.0001, decay=1e-5),
                           metrics=['accuracy'])


if __name__ == '__main__':
    squeezeNet = SqueezeNet(n_class=3)
    squeezeNet.evaluate_model(dir='train_data')