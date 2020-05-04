# yapf: disable
from keras.models import Model
from keras.layers.convolutional import Conv3D
from layers3D import *

# Use the functions provided in layers3D to build the network

def network(input_img, n_filters=16, batchnorm=True):

    # contracting path
    
    c0 = inception_block(input_img, n_filters=n_filters, batchnorm=batchnorm, strides=1, recurrent=2, layers=[[(3,1),(3,1)], [(3,2]])  # 512x512x512

    c1 = inception_block(c0, n_filters=n_filters * 2, batchnorm=batchnorm, strides=2, recurrent=2, layers=[[(3,1),(3,1)], [(3,2]])  # 256x256x256

    c2 = inception_block(c1, n_filters=n_filters * 4, batchnorm=batchnorm, strides=2, recurrent=2, layers=[[(3,1),(3,1)], [(3,2]])  # 128x128x128

    c3 = inception_block(c2, n_filters=n_filters * 8, batchnorm=batchnorm, strides=2, recurrent=2, layers=[[(3,1),(3,1)], [(3,2]])  # 64x64x64
    
    # bridge
    
    b0 = inception_block(c3, n_filters=n_filters * 16, batchnorm=batchnorm, strides=2, recurrent=2, layers=[[(3,1),(3,1)], [(3,2]])  # 32x32x32

    # expansive path
    
    attn0 = AttnGatingBlock(c3, b0, n_filters * 16)
    u0 = transpose_block(b0, attn0, n_filters=n_filters * 8)  # 64x64x64
    
    attn1 = AttnGatingBlock(c2, u0, n_filters * 8)
    u1 = transpose_block(u0, attn1, n_filters=n_filters * 4)  # 128x128x128
    
    attn2 = AttnGatingBlock(c1, u1, n_filters * 4)
    u2 = transpose_block(u1, attn2, n_filters=n_filters * 2)  # 256x256x256
    
    u3 = transpose_block(u2, c0, n_filters=n_filters)  # 512x512x512

    outputs = Conv3D(filters=1, kernel_size=1, strides=1, activation='sigmoid')(u3)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
