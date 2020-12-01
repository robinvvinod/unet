from losses import *

################################################################################
# Hyperparameters
################################################################################

# Leaky ReLU
alpha = 0.1

# Input Image Dimensions
# (rows, cols, depth, channels)
input_dimensions = (512, 512, 512, 1)
dimensions = (512, 512, 512)

# Training parameters
num_initial_filters = 32
batchnorm = True

# batch_size must be a multiple of num_gpu to ensure GPUs are not starved of data
num_gpu = 8
batch_size = 8
steps_per_epoch = 1

learning_rate = 0.00001
loss = tversky_loss
metrics = [dice_coef]
epochs = 70000

# Paths
checkpoint_path = ""
log_path = ""
save_path = ""
train_path = ""
test_path = ""
