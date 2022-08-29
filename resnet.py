import tensorflow.keras.layers as layers


"""
creates a 3*3 conv with given filters and stride
"""
def conv3x3(out_planes, stride=1):
    return layers.Conv2D(kernel_size=(3, 3), filters=out_planes, strides=stride, padding="same", use_bias=False)


"""
Creates a residual block with two 3*3 conv's
"""
basicblock_expansion = 1
bn_mom = 0.1
def basic_block(x_in, planes, stride=1, downsample=None, no_relu=False):
    residual = x_in

    x = conv3x3(planes, stride)(x_in)
    x = layers.BatchNormalization(momentum=bn_mom)(x)
    x = layers.Activation("relu")(x)

    x = conv3x3(planes,)(x)
    x = layers.BatchNormalization(momentum=bn_mom)(x)

    if downsample is not None:
        residual = downsample

    # x += residual
    x = layers.Add()([x, residual])

    if not no_relu:
        x = layers.Activation("relu")(x)

    return x


"""
creates a bottleneck block of 1*1 -> 3*3 -> 1*1
"""
bottleneck_expansion = 2
def bottleneck_block(x_in, planes, stride=1, downsample=None, no_relu=True):
    residual = x_in

    x = layers.Conv2D(filters=planes, kernel_size=(1, 1), use_bias=False)(x_in)
    x = layers.BatchNormalization(momentum=bn_mom)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters=planes, kernel_size=(3, 3), strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=bn_mom)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters=planes * bottleneck_expansion, kernel_size=(1, 1), use_bias=False)(x)
    x = layers.BatchNormalization(momentum=bn_mom)(x)

    if downsample is not None:
        residual = downsample

    # x += residual
    x = layers.Add()([x, residual])

    if not no_relu:
        x = layers.Activation("relu")(x)

    return x
