# yapf: disable
from keras.utils import Sequence
import numpy as np

class DataGenerator(Sequence):

    def __init__(self, list_IDs, labels=[], batch_size=1, dim=(512,512,512), n_channels=1, n_classes=1, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):

        # Counts the number of possible batches that can be made from the total available datasets in list_IDs
        # Rule of thumb, num_datasets % batch_size = 0, so every sample is seen
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):

        # Gets the indexes of batch_size number of data from list_IDs for one epoch
        # If batch_size = 8, 8 files/indexes from list_ID are selected
        # Makes sure that on next epoch, the batch does not come from same indexes as the previous batch
        # The same batch is not seen again until __len()__ - 1 batches are done

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

        # Creates an empty placeholder array that will be populated with data that is to be supplied
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels))

        for i, ID in enumerate(list_IDs_temp):
            # Write logic for selecting/manipulating X and y here
            pass

        return X, y
