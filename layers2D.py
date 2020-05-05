# yapf: disable
from keras.layers import BatchNormalization, Activation, Add, UpSampling2D, Concatenate, LeakyReLU
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import add, multiply, concatenate
from keras import backend as K
from hyperparameters import alpha
K.set_image_data_format('channels_last')

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, strides=1, dilation_rate=1, recurrent=1):

    # A wrapper of the Keras Conv2D block to serve as a building block for downsampling layers
    # Includes options to use batch normalization, dilation and recurrence

    conv = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides, kernel_initializer="he_normal", padding="same", dilation_rate=dilation_rate)(input_tensor)
    if batchnorm:
        conv = BatchNormalization()(conv)
    output = LeakyReLU(alpha=alpha)(conv)

    for _ in range(recurrent - 1):
        conv = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=1, kernel_initializer="he_normal", padding="same", dilation_rate=dilation_rate)(output)
        if batchnorm:
            conv = BatchNormalization()(conv)
        res = LeakyReLU(alpha=alpha)(conv)
        output = Add()([output, res])

    return output


def residual_block(input_tensor, n_filters, kernel_size=3, strides=1, batchnorm=True, recurrent=1, dilation_rate=1):

    # A residual block based on the ResNet architecture incorporating use of short-skip connections
    # Uses two successive convolution layers by default

    res = conv2d_block(input_tensor, n_filters=n_filters, kernel_size=kernel_size, strides=strides, batchnorm=batchnorm, dilation_rate=dilation_rate, recurrent=recurrent)
    res = conv2d_block(res, n_filters=n_filters, kernel_size=kernel_size, strides=1, batchnorm=batchnorm, dilation_rate=dilation_rate, recurrent=recurrent)

    shortcut = conv2d_block(input_tensor, n_filters=n_filters, kernel_size=1, strides=strides, batchnorm=batchnorm, dilation_rate=1)
    if batchnorm:
        shortcut = BatchNormalization()(shortcut)

    output = Add()([shortcut, res])
    return output


def inception_block(input_tensor, n_filters, kernel_size=3, strides=1, batchnorm=True, recurrent=1, layers=[]):

    # Inception-style convolutional block similar to InceptionNet
    # The first convolution follows the function arguments, while subsequent inception convolutions follow the parameters in
    # argument, layers

    # layers is a nested list containing the different secondary inceptions in the format of (kernel_size, dil_rate)

    # E.g => layers=[ [(3,1),(3,1)], [(5,1)], [(3,1),(3,2)] ]
    # This will implement 3 sets of secondary convolutions
    # Set 1 => 3x3 dil = 1 followed by another 3x3 dil = 1
    # Set 2 => 5x5 dil = 1
    # Set 3 => 3x3 dil = 1 followed by 3x3 dil = 2

    res = conv2d_block(input_tensor, n_filters=n_filters, kernel_size=kernel_size, strides=strides, batchnorm=batchnorm, dilation_rate=1, recurrent=recurrent)

    temp = []
    for layer in layers:
        local_res = res
        for conv in layer:
            incep_kernel_size = conv[0]
            incep_dilation_rate = conv[1]
            local_res = conv2d_block(local_res, n_filters=n_filters, kernel_size=incep_kernel_size, strides=1, batchnorm=batchnorm, dilation_rate=incep_dilation_rate, recurrent=recurrent)
        temp.append(local_res)

    temp = concatenate(temp)
    res = conv2d_block(temp, n_filters=n_filters, kernel_size=1, strides=1, batchnorm=batchnorm, dilation_rate=1)

    shortcut = conv2d_block(input_tensor, n_filters=n_filters, kernel_size=1, strides=strides, batchnorm=batchnorm, dilation_rate=1)
    if batchnorm:
        shortcut = BatchNormalization()(shortcut)

    output = Add()([shortcut, res])
    return output


def transpose_block(input_tensor, skip_tensor, n_filters, kernel_size=3, strides=1, batchnorm=True, recurrent=1):

    # A wrapper of the Keras Conv2DTranspose block to serve as a building block for upsampling layers

    shape_x = K.int_shape(input_tensor)
    shape_xskip = K.int_shape(skip_tensor)

    conv = Conv2DTranspose(filters=n_filters, kernel_size=kernel_size, padding='same', strides=(shape_xskip[1] // shape_x[1], shape_xskip[2] // shape_x[2]), kernel_initializer="he_normal")(input_tensor)
    conv = LeakyReLU(alpha=alpha)(conv)

    act = conv2d_block(conv, n_filters=n_filters, kernel_size=kernel_size, strides=1, batchnorm=batchnorm, dilation_rate=1, recurrent=recurrent)
    output = Concatenate(axis=3)([act, skip_tensor])
    return output


def expend_as(tensor, rep):

    # Anonymous lambda function to expand the specified axis by a factor of argument, rep.
    # If tensor has shape (512,512,N), lambda will return a tensor of shape (512,512,N*rep), if specified axis=2

    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
    return my_repeat


def AttnGatingBlock(x, g, inter_shape):

    shape_x = K.int_shape(x)
    shape_g = K.int_shape(g)

    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = Conv2D(filters=inter_shape, kernel_size=1, strides=1, padding='same')(g)

    # Getting the x signal to the same shape as the gating signal
    theta_x = Conv2D(filters=inter_shape, kernel_size=3, strides=(shape_x[1] // shape_g[1], shape_x[2] // shape_g[2]), padding='same')(x)

    # Element-wise addition of the gating and x signals
    add_xg = add([phi_g, theta_x])
    add_xg = Activation('relu')(add_xg)

    # 1x1x1 convolution
    psi = Conv2D(filters=1, kernel_size=1, padding='same')(add_xg)
    psi = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(psi)

    # Upsampling psi back to the original dimensions of x signal
    upsample_sigmoid_xg = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(psi)

    # Expanding the filter axis to the number of filters in the original x signal
    upsample_sigmoid_xg = expend_as(upsample_sigmoid_xg, shape_x[3])

    # Element-wise multiplication of attention coefficients back onto original x signal
    attn_coefficients = multiply([upsample_sigmoid_xg, x])

    # Final 1x1x1 convolution to consolidate attention signal to original x dimensions
    output = Conv2D(filters=shape_x[3], kernel_size=1, strides=1, padding='same')(attn_coefficients)
    output = BatchNormalization()(output)
    return output

def GatingSignal(input_tensor, batchnorm=True):

    # 1x1x1 convolution to consolidate gating signal into the required dimensions
    # Not required most of the time, unless another ReLU and batch_norm is required on gating signal

    shape = K.int_shape(input_tensor)
    conv = Conv2D(filters=shape[3], kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal")(input_tensor)
    if batchnorm:
        conv = BatchNormalization()(conv)
    output = LeakyReLU(alpha=alpha)(conv)
    return output
