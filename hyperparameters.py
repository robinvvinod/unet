from losses import *

################################################################################
# Hyperparameters
################################################################################

# Leaky ReLU
alpha = 0.1

# Input Image Dimensions
# (rows, cols, depth, channels)
input_dimensions = (512,512,512,1)
dimensions = (512,512,512)

# Dropout probability
dropout = 0.5

# Training parameters
num_initial_filters = 32
batchnorm = True
num_gpu = 8
learning_rate = 0.00001
loss = tversky_loss
metrics = [dice_coef]
epochs = 70000

# Paths
checkpoint_path = ""
log_path = ""
save_path = ""
data_path = ""
