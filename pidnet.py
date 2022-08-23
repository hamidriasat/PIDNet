# ------------------------------------------------------------------------------
# Written by Hamid Ali (hamidriasat@gmail.com)
# ------------------------------------------------------------------------------
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

bn_mom = 0.1

"""
creates a 3*3 conv with given filters and stride
"""
def conv3x3(out_planes, stride=1):
    return layers.Conv2D(kernel_size=(3,3), filters=out_planes, strides=stride, padding="same",
                       use_bias=False)

"""
Creates a residual block with two 3*3 conv's
"""
basicblock_expansion = 1
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

    x = layers.Conv2D(filters=planes, kernel_size=(1,1), use_bias=False)(x_in)
    x = layers.BatchNormalization(momentum=bn_mom)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters=planes, kernel_size=(3,3), strides=stride, padding="same",use_bias=False)(x)
    x = layers.BatchNormalization(momentum=bn_mom)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters=planes* bottleneck_expansion, kernel_size=(1,1), use_bias=False)(x)
    x= layers.BatchNormalization(momentum=bn_mom)(x)

    if downsample is not None:
        residual = downsample

    # x += residual
    x = layers.Add()([x, residual])

    if not no_relu:
        x = layers.Activation("relu")(x)

    return  x

"""
Segmentation head 
3*3 -> 1*1 -> rescale
"""
def segmentation_head(x_in, interplanes, outplanes, scale_factor=None):
    x = layers.BatchNormalization(momentum=bn_mom)(x_in)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(interplanes, kernel_size=(3, 3), use_bias=False, padding="same")(x)

    x = layers.BatchNormalization(momentum=bn_mom)(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(outplanes, kernel_size=(1, 1), use_bias=range, padding="valid")(x)  # bias difference

    if scale_factor is not None:
        input_shape = tf.keras.backend.int_shape(x)
        height2 = input_shape[1] * scale_factor
        width2 = input_shape[2] * scale_factor
        x = tf.image.resize(x, size =(height2, width2), method='bilinear')

    return x

"""
apply multiple RB or RBB blocks.
x_in: input tensor
block: block to apply it can be RB or RBB
inplanes: input tensor channels
planes: output tensor channels
blocks_num: number of time block to applied
stride: stride
expansion: expand last dimension
"""
def make_layer(x_in, block, inplanes, planes, blocks_num, stride=1, expansion=1):
    downsample = None
    if stride != 1 or inplanes != planes * expansion:
        downsample = layers.Conv2D((planes * expansion), kernel_size=(1, 1),strides=stride, use_bias=False)(x_in)
        downsample = layers.BatchNormalization(momentum=bn_mom)(downsample)
        # In original resnet paper relu was applied, But in pidenet it was not used
        # So commenting out for now
        # downsample = layers.Activation("relu")(downsample)

    x = block(x_in, planes, stride, downsample)
    for i in range(1, blocks_num):
        if i == (blocks_num - 1):
            x = block(x, planes, stride=1, no_relu=True)
        else:
            x = block(x, planes, stride=1, no_relu=False)

    return x

# Deep Aggregation Pyramid Pooling Module
def DAPPPM(x_in, branch_planes, outplanes):
    input_shape = tf.keras.backend.int_shape(x_in)
    height = input_shape[1]
    width = input_shape[2]
    # Average pooling kernel size
    kernal_sizes_height = [5, 9, 17, height]
    kernal_sizes_width =  [5, 9, 17, width]
    # Average pooling strides size
    stride_sizes_height = [2, 4, 8, height]
    stride_sizes_width =  [2, 4, 8, width]
    x_list = []

    # y1
    scale0 = layers.BatchNormalization(momentum=bn_mom)(x_in)
    scale0 = layers.Activation("relu")(scale0)
    scale0 = layers.Conv2D(branch_planes, kernel_size=(1,1), use_bias=False, )(scale0)
    x_list.append(scale0)

    for i in range( len(kernal_sizes_height)):
        # first apply average pooling
        temp = layers.AveragePooling2D(pool_size=(kernal_sizes_height[i],kernal_sizes_width[i]),
                                       strides=(stride_sizes_height[i],stride_sizes_width[i]),
                                       padding="same")(x_in)
        temp = layers.BatchNormalization(momentum=bn_mom)(temp)
        temp = layers.Activation("relu")(temp)
        # then apply 1*1 conv
        temp = layers.Conv2D(branch_planes, kernel_size=(1, 1), use_bias=False, )(temp)
        # then resize using bilinear
        temp = tf.image.resize(temp, size=(height,width), )
        # add current and previous layer output
        temp = layers.Add()([temp, x_list[i]])
        temp = layers.BatchNormalization(momentum=bn_mom)(temp)
        temp = layers.Activation("relu")(temp)
        # at the end apply 3*3 conv
        temp = layers.Conv2D(branch_planes, kernel_size=(3, 3), use_bias=False, padding="same")(temp)
        # y[i+1]
        x_list.append(temp)

    # concatenate all
    combined = layers.concatenate(x_list, axis=-1)

    combined = layers.BatchNormalization(momentum=bn_mom)(combined)
    combined = layers.Activation("relu")(combined)
    combined = layers.Conv2D(outplanes, kernel_size=(1, 1), use_bias=False, )(combined)

    shortcut = layers.BatchNormalization(momentum=bn_mom)(x_in)
    shortcut = layers.Activation("relu")(shortcut)
    shortcut = layers.Conv2D(outplanes, kernel_size=(1, 1), use_bias=False, )(shortcut)

    # final = combined + shortcut
    final = layers.Add()([combined, shortcut])

    return final


# Parallel Aggregation Pyramid Pooling Module
def PAPPM(x_in, branch_planes, outplanes):
    input_shape = tf.keras.backend.int_shape(x_in)
    height = input_shape[1]
    width = input_shape[2]
    # Average pooling kernel size
    kernal_sizes_height = [5, 9, 17, height]
    kernal_sizes_width =  [5, 9, 17, width]
    # Average pooling strides size
    stride_sizes_height = [2, 4, 8, height]
    stride_sizes_width =  [2, 4, 8, width]
    x_list = []

    scale0 = layers.BatchNormalization(momentum=bn_mom)(x_in)
    scale0 = layers.Activation("relu")(scale0)
    scale0 = layers.Conv2D(branch_planes, kernel_size=(1,1), use_bias=False, )(scale0)

    for i in range( len(kernal_sizes_height)):
        # first apply average pooling
        temp = layers.AveragePooling2D(pool_size=(kernal_sizes_height[i],kernal_sizes_width[i]),
                                       strides=(stride_sizes_height[i],stride_sizes_width[i]),
                                       padding="same")(x_in)
        temp = layers.BatchNormalization(momentum=bn_mom)(temp)
        temp = layers.Activation("relu")(temp)
        # then apply 1*1 conv
        temp = layers.Conv2D(branch_planes, kernel_size=(1, 1), use_bias=False, )(temp)
        # then resize using bilinear
        temp = tf.image.resize(temp, size=(height,width), method = 'bilinear')
        temp = layers.Add()([temp, scale0])

        x_list.append(temp)

    # concatenate all
    combined = layers.concatenate(x_list, axis=-1)

    # scale_out
    combined = layers.BatchNormalization(momentum=bn_mom)(combined)
    combined = layers.Activation("relu")(combined)
    combined = layers.Conv2D(branch_planes * 4 , kernel_size=(3, 3), use_bias=False, padding="same", groups=4)(combined)

    # concatenate all
    combined = layers.concatenate([scale0, combined], axis=-1)

    # compression
    combined = layers.BatchNormalization(momentum=bn_mom)(combined)
    combined = layers.Activation("relu")(combined)
    combined = layers.Conv2D(outplanes, kernel_size=(1, 1), use_bias=False, )(combined)

    # shortcut
    shortcut = layers.BatchNormalization(momentum=bn_mom)(x_in)
    shortcut = layers.Activation("relu")(shortcut)
    shortcut = layers.Conv2D(outplanes, kernel_size=(1, 1), use_bias=False, )(shortcut)

    # final = combined + shortcut
    final = layers.Add()([combined, shortcut])

    return final


# Pixel-attention-guided fusion module
def PagFM(x_in, y_in, in_planes, mid_planes, after_relu=False, with_planes=False):
    x_shape = tf.keras.backend.int_shape(x_in)
    if after_relu:
        x_in = layers.Activation("relu")(x_in)
        y_in = layers.Activation("relu")(y_in)

    y_q = layers.Conv2D(mid_planes, kernel_size=(1, 1), use_bias=False)(y_in)
    y_q = layers.BatchNormalization(momentum=bn_mom)(y_q)
    y_q = tf.image.resize(y_q, size=(x_shape[1], x_shape[2]), method='bilinear')

    x_k = layers.Conv2D(mid_planes, kernel_size=(1, 1), use_bias=False)(x_in)
    x_k = layers.BatchNormalization(momentum=bn_mom)(x_k)

    if with_planes:
        sim_map = x_k * y_q
        sim_map = layers.Conv2D(in_planes, kernel_size=(1, 1), use_bias=False)(sim_map)
        sim_map = layers.BatchNormalization(momentum=bn_mom)(sim_map)
        sim_map = layers.Activation("sigmoid")(sim_map)
    else:
        sim_map = x_k * y_q
        sim_map = tf.math.reduce_sum(sim_map, axis=-1, keepdims=True)
        sim_map = layers.Activation("sigmoid")(sim_map)

    y_in = tf.image.resize(y_in, size=(x_shape[1], x_shape[2]), method='bilinear')
    x_in = (1 - sim_map) * x_in + sim_map * y_in
    return x_in


# Light Boundary-attention-guided
def Light_Bag(p, i, d, planes):
    edge_att = layers.Activation("sigmoid")(d)

    p_in = (1-edge_att) * i + p
    i_in = i + edge_att * p

    p_add = layers.Conv2D(filters=planes, kernel_size=(1, 1), use_bias=False)(p_in)
    p_add = layers.BatchNormalization(momentum=bn_mom)(p_add)

    i_add = layers.Conv2D(filters=planes, kernel_size=(1, 1), use_bias=False)(i_in)
    i_add = layers.BatchNormalization(momentum=bn_mom)(i_add)

    # final = combined + shortcut
    final = layers.Add()([p_add, i_add])
    return final

# Boundary-attention-guided
def Bag(p, i, d, planes):
    edge_att = layers.Activation("sigmoid")(d)
    x = edge_att * p + (1 - edge_att) * i

    x = layers.BatchNormalization(momentum=bn_mom)(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(planes, kernel_size=(3, 3), padding="same", use_bias=False)(x)

    return x


def PIDNet(input_shape=[1024,2048,3], m=2, n=3, num_classes=19, planes=64, ppm_planes=96,
           head_planes=128, augment=True):

    x_in = layers.Input(input_shape)

    input_shape = tf.keras.backend.int_shape(x_in)
    height_output = input_shape[1] // 8
    width_output = input_shape[2] // 8

    # I Branch
    x = layers.Conv2D(planes, kernel_size=(3, 3), strides=2, padding='same')(x_in)
    x = layers.BatchNormalization(momentum=bn_mom)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(planes, kernel_size=(3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization(momentum=bn_mom)(x)
    x = layers.Activation("relu")(x)

    x = make_layer(x, basic_block, planes, planes, m, expansion=basicblock_expansion)  # layer1
    x = layers.Activation("relu")(x)

    x = make_layer(x, basic_block, planes, planes * 2, m, stride=2, expansion=basicblock_expansion)  # layer2
    x = layers.Activation("relu")(x)

    x_ = make_layer(x, basic_block, planes * 2, planes * 2, m, expansion=basicblock_expansion)  # layer3_
    if m == 2:
        x_d = make_layer(x, basic_block, planes * 2, planes, 0, expansion=basicblock_expansion)  # layer3_d
    else:
        x_d = make_layer(x, basic_block, planes * 2, planes * 2, 0, expansion=basicblock_expansion)  # layer3_d
    x_d = layers.Activation("relu")(x_d)

    x = make_layer(x, basic_block, planes * 2, planes * 4, n, stride=2, expansion=basicblock_expansion)  # layer3
    x = layers.Activation("relu")(x)

    # P Branch
    compression3 = layers.Conv2D(planes * 2, kernel_size=(1, 1), use_bias=False)(x)  # compression3
    compression3 = layers.BatchNormalization(momentum=bn_mom)(compression3)

    x_ = PagFM(x_, compression3, planes * 2, planes)  # pag3

    if m == 2:
        diff3 = layers.Conv2D(planes, kernel_size=(3, 3), padding='same', use_bias=False)(x)  # diff3
        diff3 = layers.BatchNormalization(momentum=bn_mom)(diff3)
    else:
        diff3 = layers.Conv2D(planes * 2, kernel_size=(3, 3), padding='same', use_bias=False)(x)  # diff3
        diff3 = layers.BatchNormalization(momentum=bn_mom)(diff3)

    diff3 = tf.image.resize(diff3, size=(height_output, width_output), method='bilinear')
    x_d = x_d + diff3

    if augment:
        temp_p = x_

    layer4 = make_layer(x, basic_block, planes * 4, planes * 8, n, stride=2, expansion=basicblock_expansion)  # layer4
    x = layers.Activation("relu")(layer4)

    x_ = layers.Activation("relu")(x_)
    x_ = make_layer(x_, basic_block, planes * 2, planes * 2, m, expansion=basicblock_expansion)  # layer4_

    x_d = layers.Activation("relu")(x_d)
    if m == 2:
        x_d = make_layer(x_d, bottleneck_block, planes, planes, 1, expansion=bottleneck_expansion)  # layer4_d
    else:
        x_d = make_layer(x_d, basic_block, planes * 2, planes * 2, 0, expansion=basicblock_expansion)  # layer4_d
        x_d = layers.Activation("relu")(x_d)

    compression4 = layers.Conv2D(planes * 2, kernel_size=(1, 1), use_bias=False)(x)  # compression4
    compression4 = layers.BatchNormalization(momentum=bn_mom)(compression4)
    x_ = PagFM(x_, compression4, planes * 2, planes)  # pag4

    diff4 = layers.Conv2D(planes * 2, kernel_size=(3, 3), padding='same', use_bias=False)(x)  # diff4
    diff4 = layers.BatchNormalization(momentum=bn_mom)(diff4)
    diff4 = tf.image.resize(diff4, size=(height_output, width_output), method='bilinear')
    x_d = x_d + diff4

    if augment:
        temp_d = x_d

    x_ = layers.Activation("relu")(x_)
    x_ = make_layer(x_, bottleneck_block, planes * 2, planes * 2, 1, expansion=bottleneck_expansion)  # layer5_

    x_d = layers.Activation("relu")(x_d)
    x_d = make_layer(x_d, bottleneck_block, planes * 2, planes * 2, 1, expansion=bottleneck_expansion)  # layer5_d

    layer5 = make_layer(x, bottleneck_block, planes * 8, planes * 8, 2, stride=2, expansion=bottleneck_expansion)  # layer5
    if m == 2:
        spp = PAPPM(layer5, ppm_planes, planes * 4)  # spp
        x = tf.image.resize(spp, size=(height_output, width_output), method='bilinear')
        dfm = Light_Bag(x_, x, x_d, planes * 4)  # dfm
    else:
        spp = DAPPPM(layer5,  ppm_planes, planes * 4)  # spp
        x = tf.image.resize(spp, size=(height_output, width_output), method='bilinear')
        dfm = Bag(x_, x, x_d, planes * 4)  # dfm

    x_ = segmentation_head(dfm, head_planes, num_classes)  # final_layer

    # Prediction Head
    if augment:
        seghead_p = segmentation_head(temp_p, head_planes, num_classes)
        seghead_d = segmentation_head(temp_d, planes, 1)
        model_output = [seghead_p, x_, seghead_d]
    else:
        model_output = [x_]

            # x = x_in
    # x = PagFM(x, x[:, :64, :128, :], planes * 4, planes)
    # x = PAPPM(x,  ppm_planes, planes * 4)  # For PAPPM
    # model_output = x

    model = models.Model(inputs=[x_in], outputs=model_output)

    # set weight initializers
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel_initializer = tf.keras.initializers.he_normal()
        if hasattr(layer, 'beta_initializer'):  # for BatchNormalization
            layer.beta_initializer = "zeros"
            layer.gamma_initializer = "ones"


    return model


"""
create PIDNet
name : pidnet_s for small version, pidnet_m for medium, pidnet for  version
input_shape : shape of input data
num_classes: output classes
"""
def get_pred_model(name, input_shape, num_classes):

    # small
    if 's' in name:
        model = PIDNet(input_shape=input_shape, m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=False)
    elif 'm' in name:
        model = PIDNet(input_shape=input_shape,m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=False)
    else:
        model = PIDNet(input_shape=input_shape,m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=False)

    return model


if __name__ == "__main__":
    """## Model Compilation"""
    print("Initializing Model")
    INPUT_SHAPE = [1024, 2048, 3]
    OUTPUT_CHANNELS = 19
    with tf.device("cpu:0"):
        # create model
        pidnet_model = get_pred_model("pidnet_s", INPUT_SHAPE, OUTPUT_CHANNELS)
        optimizer = tf.keras.optimizers.SGD(momentum=0.9, lr=0.045)
        # compile model
        pidnet_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), optimizer=optimizer,
                          metrics=['accuracy'])
        # show model summary in output
        pidnet_model.summary()

        print("Done")

