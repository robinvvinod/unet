# yapf: disable
from keras.models import load_model
from hyperparameters import *
from datagenerator import DataGenerator
import numpy as np
from losses import *

model = load_model(save_path)

list_IDs = []
for filename in os.listdir(test_path):
    # Write logic to add filenames of train images to list_IDs which will be processed by DataGenerator
    # later on
    pass

evaluate_gen = DataGenerator(list_IDs=list_IDs, dim=dimensions, batch_size=batch_size, shuffle=True)

# Returns Numpy arrays of predictions
model.predict_generator(evaluate_gen, steps=0, verbose=2, workers=20)
