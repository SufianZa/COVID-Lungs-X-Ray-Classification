from tensorflow.keras.utils import Sequence
import numpy as np
import os
import cv2

class CustomImageGenerator(Sequence):
    def __init__(self, files, labels, directory, n_features=2, batch_size=16, img_dim=(384, 384, 1),
                 n_classes=3, shuffle=True):
        self.img_dim = img_dim
        self.files = files
        self.n = len(self.files)
        self.labels = labels
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.n_features = n_features
        self.shuffle = shuffle
        self.dir = directory
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.files))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        files_batch = [self.files[k] for k in indices]
        labels_batch = np.array([self.labels[k] for k in indices])
        return self.get_images_batch(files_batch), labels_batch

    def get_images_batch(self, files_batch):
        images = np.empty((self.batch_size, *self.img_dim))
        # features = np.zeros((self.batch_size, self.n_features))
        for i, file_name in enumerate(files_batch):
            img = np.array(cv2.imread(os.path.join(self.dir, file_name), cv2.IMREAD_GRAYSCALE)).astype(float)
            images[i, :, :, 0] = (img - np.min(img)) / (np.max(img) - np.min(img))
        return images
