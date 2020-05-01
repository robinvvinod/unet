# yapf: disable
from keras.models import Model
from keras.layers.convolutional import Conv3D
from keras.layers import SpatialDropout3D
from layers3D import *

# Use the functions provided in layers3D to build the network

def network(input_img, n_filters=16, dropout=0.5, batchnorm=True):

    # contracting path
    
    c0 = inception_block(input_img, n_filters=n_filters, batchnorm=batchnorm, strides=1, recurrent=2, layers=((3,2),(5,1)))  # 512x512x512
    p0 = SpatialDropout3D(dropout * 0.5)(c0)

    c1 = inception_block(p0, n_filters=n_filters * 2, batchnorm=batchnorm, strides=2, recurrent=2, layers=((3,2),(5,1)))  # 256x256x256
    p1 = SpatialDropout3D(dropout)(c1)

    c2 = inception_block(p1, n_filters=n_filters * 4, batchnorm=batchnorm, strides=2, recurrent=2, layers=((3,2),(5,1)))  # 128x128x128
    p2 = SpatialDropout3D(dropout)(c2)

    c3 = inception_block(p2, n_filters=n_filters * 8, batchnorm=batchnorm, strides=2, recurrent=2, layers=((3,2),(5,1)))  # 64x64x64
    p3 = SpatialDropout3D(dropout)(c3)
    
    # bridge
    
    b0 = inception_block(p3, n_filters=n_filters * 16, batchnorm=batchnorm, strides=2, recurrent=2, layers=((3,2),(5,1)))  # 32x32x32

    # expansive path
    
    attn0 = AttnGatingBlock(p3, b0, n_filters * 16)
    u0 = transpose_block(b0, attn0, n_filters=n_filters * 8)  # 64x64x64
    d0 = SpatialDropout3D(dropout)(u0)
    
    attn1 = AttnGatingBlock(p2, d0, n_filters * 8)
    u1 = transpose_block(d0, attn1, n_filters=n_filters * 4)  # 128x128x128
    d1 = SpatialDropout3D(dropout)(u1)
    
    attn2 = AttnGatingBlock(p1, d1, n_filters * 4)
    u2 = transpose_block(d1, attn2, n_filters=n_filters * 2)  # 256x256x256
    d2 = SpatialDropout3D(dropout)(u2)
    
    u3 = transpose_block(d2, p0, n_filters=n_filters)  # 512x512x512
    d3 = SpatialDropout3D(dropout)(u3)

    outputs = Conv3D(filters=1, kernel_size=1, strides=1, activation='sigmoid')(d3)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
