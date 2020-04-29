# Keras U-Net

## What is it?

Keras implementation of a 2D/3D U-Net with the following implementations provided:
* Additive attention -- [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)
* Inception convolutions w/ dilated convolutions -- [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) and [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122)
* Recurrent convolutions -- [R2U-Net](https://arxiv.org/abs/1802.06955)
* Focal Tversky Loss
* Dice Coefficient Loss

## Usage

### Dependencies

This repository depends on the following libraries:
* Tensorflow
* Keras
* Python 3
* Numpy
* Matplotlib

### Building your network

The pre-implemented layers are available in [`layers3D.py`](layers3D.py). Use the layers to build your preferred network configuration in [`network.py`](network.py)

##### Example

```
from layers3D import *
from keras.models import Model

def network(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    outputs = inception_block(input_img, n_filters=n_filters, batchnorm=batchnorm, strides=1, recurrent=2)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
```
*Refer to [`network.py`](network.py) for a full example*

### Data Generator

Rewrite the `__data_generation()` method in [`datagenerator.py`](datagenerator.py) to supply batches of data during training

##### Example

```
def __data_generation(self, list_IDs_temp):

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels))

        for i, ID in enumerate(list_IDs_temp):
            # Write logic for selecting/manipulating X and y here
            X[i,] = np.load('path/to/x/ID')
            y[i,] = np.load('path/to/y/ID')

        return X, y
```

The `DataGenerator` class in [`train.py`](train.py) takes in `list` arguments containing the ID (filenames) of X and y

### Hyperparameters

Set the appropriate values for the hyper-parameters listed in [`hyperparameters.py`](hyperparameters.py)

### Train

Run [`train.py`](train.py) once all the configuration is done to train your network

### Testing

Run [`evaluate.py`](evaluate.py) or [`predict.py`](predict.py) with the appropriate list_IDs provided to the `DataGenerator`

