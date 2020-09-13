from tensorflow.keras.layers import Dropout, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import applications
from Models.BaseModel import BaseModel


class ResNet152(BaseModel):
    def __init__(self, input_size=(224, 224, 3), n_class=3, batch_size=16, epochs=60):
        super().__init__(input_size, n_class, batch_size, epochs, model_name='ResNet152')

    def init_network(self):
        model = applications.ResNet152(include_top=False, weights='imagenet',
                                       input_shape=self.input_size)
        x = model.output
        x = AveragePooling2D(pool_size=(4, 4))(x)
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        output = Dense(len(self.CLASS_TARGETS), activation="softmax")(x)
        for layer in model.layers:
            layer.trainable = False

        # create Model
        self.model = Model(inputs=model.inputs, outputs=output)

        # compile Model
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])