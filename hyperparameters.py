from losses import *

################################################################################
# Hyperparameters
################################################################################

# Leaky ReLU
alpha = 0.1

# Input Image Dimensions
dimensions = (512,512,512)

# Dropout probability
dropout = 0.5

# Training parameters
num_initial_filters = 32
batchnorm = True
num_gpu = 8
learning_rate = 0.00001
loss = tversky
metrics = ['dice_coef']
epochs = 70000

# Paths
checkpoint_path = ''
log_path = ''
save_path = ''
