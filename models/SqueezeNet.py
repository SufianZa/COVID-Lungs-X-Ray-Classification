from tensorflow.keras.layers import Conv2D, MaxPool2D, concatenate, \
    GlobalAveragePooling2D, Dropout, Input, Activation, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from models.BaseModel import BaseModel


class SqueezeNet(BaseModel):
    def __init__(self, input_size=(320, 320, 1), n_class=3, batch_size=16, epochs=60):
        super().__init__(input_size, n_class, batch_size, epochs,  model_name='SqueezeNet')

    def fire(self, x, squeeze_size):
        return self.expand(self.squeeze(x, squeeze_size), squeeze_size * 4)

    def squeeze(self, y, squeeze_size):
        return Conv2D(filters=squeeze_size, kernel_size=1, activation='relu', padding='same')(y)

    def expand(self, x, expand_size):
        left = Conv2D(filters=expand_size, kernel_size=1, activation='relu', padding='same')(x)
        right = Conv2D(filters=expand_size, kernel_size=3, activation='relu', padding='same')(x)
        return concatenate([left, right], axis=3)

    def init_network(self):
        # input
        input_layer = Input(self.input_size)
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