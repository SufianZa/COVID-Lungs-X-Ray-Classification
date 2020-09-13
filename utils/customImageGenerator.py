from tensorflow.keras.utils import Sequence
import numpy as np
import os
import cv2


class CustomImageGenerator(Sequence):
    def __init__(self, files, labels, directory, n_features=2, batch_size=16, input_size=(320,320,1),
                 n_classes=3, shuffle=True):
        self.img_dim = input_size[:2]
        self.channels = input_size[2]
        self.files = files
        self.n = len(self.files)
        self.labels = labels
        self.batch_size = batch_size
        self.resize = self.img_dim != (320, 320)
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

    def cut_edges(self, img, top=0, bottom=0, left=0, right=0, a=0):
      w,h = img.shape
      img = np.array(img)
      return img[top+a:h-bottom-a,left+a:w-right-a]

    def get_images_batch(self, files_batch):
        images = np.empty((self.batch_size, *self.img_dim, self.channels))
        for i, file_name in enumerate(files_batch):
            file_path = os.path.join(self.dir, file_name)
            if not (os.path.isfile(file_path) and os.access(file_path, os.R_OK)):
                print("Error: Either the file is missing or not readable {}".format(file_name))
                continue
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = self.cut_edges(img, 25, 25, 25, 25)
            if self.resize or 1:
                img = np.array(cv2.resize(img, dsize=self.img_dim, interpolation=cv2.INTER_CUBIC)).astype(float)
            for c in range(self.channels):
                images[i, :, :, c] = np.array(img)/255
        return images
