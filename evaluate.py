# yapf: disable
from keras.models import load_model
from hyperparameters import save_path, dimensions, num_gpu
from datagenerator import DataGenerator
from losses import *

model = load_model(save_path)
evaluate_gen = DataGenerator(list_IDs=[], labels=[], dim=dimensions, batch_size=num_gpu, shuffle=True)

# Returns test loss using metric specified in train.py during training
model.evaluate_generator(evaluate_gen, steps=0, verbose=2, workers=20)
