from tensorflow.keras.layers import Conv2D, MaxPool2D, concatenate, \
    GlobalAveragePooling2D, Dropout, Input, Activation, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import applications
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


class BaseModel:
    def __init__(self, input_size=(384, 384, 1), n_class=3, model_name='undefined'):
        self.model_name = model_name
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

        # split data into 80% train, 15% validation and 5% unseen test data
        n_samples = len(df)
        train_x, test_x, train_y, test_y = train_test_split(df[0].tolist(), df[3].tolist(), test_size=0.2,
                                                            random_state=5855, stratify=df[3].tolist())
        val_x, unseen_x, val_y, unseen_y = train_test_split(test_x, test_y, test_size=0.25, random_state=4255,
                                                            stratify=test_y)
        print('Number of samples: {} --- {}% train data | {}% validation data | {}% test data'.format(n_samples, (
                len(train_x) / n_samples) * 100, (len(val_x) / n_samples) * 100, (len(unseen_x) / n_samples) * 100))

        # create generators
        train_generator = CustomImageGenerator(train_x, train_y, directory=dir, batch_size=32,
                                               input_size=self.input_size)
        valid_generator = CustomImageGenerator(val_x, val_y, directory=dir, batch_size=32, input_size=self.input_size)

        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=8,
                                   verbose=0, mode='auto')
        # fit model
        self.history = self.model.fit(train_generator, validation_data=valid_generator, epochs=1,
                                      steps_per_epoch=train_generator.n // train_generator.batch_size,
                                      validation_steps=valid_generator.n // valid_generator.batch_size,
                                      callbacks=[early_stop])

        # save weight and test data
        self.model.save_weights(self.weight_file)
        with open('unseen_data.pkl', "wb") as f:
            pickle.dump({'x': unseen_x, 'y': unseen_y, 'history': self.history.history}, f)

    def evaluate_model(self, dir):
        self.model.load_weights(self.weight_file)
        with open("unseen_data.pkl", "rb") as f:
            data = pickle.load(f)
        # data['y'] = [data['y'][idx] for idx, f in enumerate(data['x']) if os.path.isfile(os.path.join(dir, f)) and os.access(os.path.join(dir, f), os.R_OK)]
        # data['x'] = [f for idx, f in enumerate(data['x']) if os.path.isfile(os.path.join(dir, f)) and os.access(os.path.join(dir, f), os.R_OK)]
        unseen_gen = CustomImageGenerator(data['x'], data['y'], directory=dir, batch_size=1, shuffle=False,
                                          input_size=self.input_size)
        predictions = self.model.predict(unseen_gen, workers=0, use_multiprocessing=False, verbose=0)

        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(unseen_gen.labels, axis=1)

        show_confusion_Matrix(y_pred, y_true, self.CLASS_TARGETS, self.model_name)
        # plt.savefig('conf_matrix.png')

        show_learning_curves(data['history'], self.model_name)
        # plt.savefig('learning_curve.png')

        plt.show()


class SqueezeNet(BaseModel):
    def fire(self, x, squeeze_size):
        return self.expand(self.squeeze(x, squeeze_size), squeeze_size * 4)

    def squeeze(self, y, squeeze_size):
        return Conv2D(filters=squeeze_size, kernel_size=1, activation='relu', padding='same')(y)

    def expand(self, x, expand_size):
        left = Conv2D(filters=expand_size, kernel_size=1, activation='relu', padding='same')(x)
        right = Conv2D(filters=expand_size, kernel_size=3, activation='relu', padding='same')(x)
        return concatenate([left, right], axis=3)

    def __init__(self, input_size=384, n_class=3):
        super().__init__(input_size, n_class, model_name='SqueezeNet')

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


class ResNet152(BaseModel):
    def __init__(self, input_size=(224, 224, 3), n_class=3):
        super().__init__(input_size, n_class, model_name='ResNet50V2')

    def init_network(self):
        model = applications.ResNet152(include_top=False, weights='imagenet',
                                       input_shape=self.input_size)
        x = model.output
        x = AveragePooling2D(pool_size=(4, 4))(x)
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        output = Dense(3, activation="softmax")(x)
        for layer in model.layers:
            layer.trainable = False

        # create Model
        self.model = Model(inputs=model.inputs, outputs=output)

        # compile Model
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(lr=0.0001, decay=1e-5),
                           metrics=['accuracy'])


class VGG16(BaseModel):
    def __init__(self, input_size=(224, 224, 3), n_class=3):
        super().__init__(input_size, n_class, model_name='Vgg16')

    def init_network(self):
        model = applications.vgg16(include_top=False, weights='imagenet',
                                   input_shape=self.input_size)
        x = model.output
        x = AveragePooling2D(pool_size=(4, 4))(x)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.5)(x)
        output = Dense(3, activation="softmax")(x)
        for layer in model.layers:
            layer.trainable = False

        # create Model
        self.model = Model(inputs=model.inputs, outputs=output)

        # compile Model
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])


class DenseNet201(BaseModel):
    def __init__(self, input_size=(224, 224, 3), n_class=3):
        super().__init__(input_size, n_class, model_name='DenseNet201')

    def init_network(self):
        model = applications.DenseNet201(include_top=False, weights='imagenet',
                                         input_shape=self.input_size)
        x = model.output
        x = AveragePooling2D(pool_size=(4, 4))(x)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.5)(x)
        output = Dense(3, activation="softmax")(x)
        for layer in model.layers:
            layer.trainable = False

        # create Model
        self.model = Model(inputs=model.inputs, outputs=output)

        # compile Model
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])


if __name__ == '__main__':
    squeezeNet = ResNet152(n_class=3)
    squeezeNet.train(dir='train_data', pkl_file='paths_features_labels.pkl')
    squeezeNet.evaluate_model(dir='train_data')
