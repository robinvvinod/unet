# yapf: disable
from keras.layers import BatchNormalization, Activation, Add, UpSampling3D, Concatenate, LeakyReLU
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.merge import add, multiply, concatenate
from keras import backend as K
from hyperparameters import alpha
K.set_image_data_format('channels_last')

def conv3d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, strides=1, dilation_rate=1, recurrent=1):

    # A wrapper of the Keras Conv3D block to serve as a building block for downsampling layers
    # Includes options to use batch normalization, dilation and recurrence

    conv = Conv3D(filters=n_filters, kernel_size=kernel_size, strides=strides, kernel_initializer="he_normal", padding="same", dilation_rate=dilation_rate)(input_tensor)
    if batchnorm:
        conv = BatchNormalization()(conv)
    output = LeakyReLU(alpha=alpha)(conv)

    for _ in range(recurrent - 1):
        conv = Conv3D(filters=n_filters, kernel_size=kernel_size, strides=1, kernel_initializer="he_normal", padding="same", dilation_rate=dilation_rate)(output)
        if batchnorm:
            conv = BatchNormalization()(conv)
        res = LeakyReLU(alpha=alpha)(conv)
        output = Add()([output, res])

    return output


def residual_block(input_tensor, n_filters, kernel_size=3, strides=1, batchnorm=True, recurrent=1, dilation_rate=1):

    # A residual block based on the ResNet architecture incorporating use of short-skip connections
    # Uses two successive convolution layers by default

    res = conv3d_block(input_tensor, n_filters=n_filters, kernel_size=kernel_size, strides=strides, batchnorm=batchnorm, dilation_rate=dilation_rate, recurrent=recurrent)
    res = conv3d_block(res, n_filters=n_filters, kernel_size=kernel_size, strides=1, batchnorm=batchnorm, dilation_rate=dilation_rate, recurrent=recurrent)

    shortcut = conv3d_block(input_tensor, n_filters=n_filters, kernel_size=1, strides=strides, batchnorm=batchnorm, dilation_rate=1)
    if batchnorm:
        shortcut = BatchNormalization()(shortcut)

    output = Add()([shortcut, res])
    return output


def inception_block(input_tensor, n_filters, kernel_size=3, strides=1, batchnorm=True, recurrent=1, layers=()):

    # Inception-style convolutional block similar to InceptionNet
    # The first convolution follows the function arguments, while subsequent inception convolutions follow the parameters in
    # argument, layers

    # layers is a tuple containing the different kernel_sizes and dilation rates of the secondary inception convolutions
    # E.g => layers=( ((3,3,3),(2,2,2)), ((5,5,5),1), (7,1) )
    # This will implement 3 convolutions of kernel_sizes 3x3x3, 5x5x5, 7x7x7 with respective dilation rates of 2x2x2,
    # 1x1x1 and 1x1x1

    res = conv3d_block(input_tensor, n_filters=n_filters, kernel_size=kernel_size, strides=strides, batchnorm=batchnorm, dilation_rate=1, recurrent=recurrent)

    temp = []
    for conv in layers:
        incep_kernel_size = conv[0]
        incep_dilation_rate = conv[1]
        temp.append(conv3d_block(res, n_filters=n_filters, kernel_size=incep_kernel_size, strides=1, batchnorm=batchnorm, dilation_rate=incep_dilation_rate, recurrent=recurrent))

    temp = concatenate(temp)
    res = conv3d_block(temp, n_filters=n_filters, kernel_size=1, strides=1, batchnorm=batchnorm, dilation_rate=1, recurrent=recurrent)

    shortcut = conv3d_block(input_tensor, n_filters=n_filters, kernel_size=1, strides=strides, batchnorm=batchnorm, dilation_rate=1)
    if batchnorm:
        shortcut = BatchNormalization()(shortcut)

    output = Add()([shortcut, res])
    return output


def transpose_block(input_tensor, skip_tensor, n_filters, kernel_size=3, strides=1):

    # A wrapper of the Keras Conv3DTranspose block to serve as a building block for upsampling layers

    shape_x = K.int_shape(input_tensor)
    shape_xskip = K.int_shape(skip_tensor)

    conv = Conv3DTranspose(filters=n_filters, kernel_size=kernel_size, padding='same', strides=(shape_xskip[1] // shape_x[1], shape_xskip[2] // shape_x[2], shape_xskip[3] // shape_x[3]), kernel_initializer="he_normal")(input_tensor)
    act = LeakyReLU(alpha=alpha)(conv)
    output = Concatenate(axis=4)([act, skip_tensor])
    return output


def expend_as(tensor, rep):

    # Anonymous lambda function to expand the specified axis by a factor of argument, rep.
    # If tensor has shape (512,512,N), lambda will return a tensor of shape (512,512,N*rep), if specified axis=2

    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=4), arguments={'repnum': rep})(tensor)
    return my_repeat


def AttnGatingBlock(x, g, inter_shape):

    shape_x = K.int_shape(x)
    shape_g = K.int_shape(g)

    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = Conv3D(filters=inter_shape, kernel_size=(1,1,1), strides=(1,1,1), padding='same')(g)

    # Getting the x signal to the same shape as the gating signal
    theta_x = Conv3D(filters=inter_shape, kernel_size=(3,3,3), strides=(shape_x[1] // shape_g[1], shape_x[2] // shape_g[2], shape_x[3] // shape_g[3]), padding='same')(x)

    # TODO Is this layer necessary?
    # shape_theta_x = K.int_shape(theta_x)
    # upsample_phi_g = Conv3DTranspose(filters=inter_shape, kernel_size=(3,3,3), strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2], shape_theta_x[3] // shape_g[3]), padding='same')(phi_g)

    # Element-wise addition of the gating and x signals
    add_xg = add([phi_g, theta_x])
    add_xg = Activation('relu')(add_xg)

    # 1x1x1 convolution
    psi = Conv3D(filters=1, kernel_size=1, padding='same')(add_xg)
    psi = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(psi)

    # Upsampling psi back to the original dimensions of x signal
    upsample_sigmoid_xg = UpSampling3D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3]))(psi)

    # Expanding the filter axis to the number of filters in the original x signal
    upsample_sigmoid_xg = expend_as(upsample_sigmoid_xg, shape_x[4])

    # Element-wise multiplication of attention coefficients back onto original x signal
    attn_coefficients = multiply([upsample_sigmoid_xg, x])

    # Final 1x1x1 convolution to consolidate attention signal to original x dimensions
    output = Conv3D(filters=shape_x[4], kernel_size=(1,1,1), strides=(1,1,1), padding='same')(attn_coefficients)
    output = BatchNormalization()(output)
    return output

def UnetGatingSignal(input_tensor, batchnorm=True):

    # 1x1x1 convolution to consolidate gating signal into the required dimensions

    shape = K.int_shape(input_tensor)
    conv = Conv3D(filters=shape[4], kernel_size=(1,1,1), strides=(1,1,1), padding="same", kernel_initializer="he_normal")(input_tensor)
    if batchnorm:
        conv = BatchNormalization()(conv)
    output = LeakyReLU(alpha=alpha)(conv)
    return output
