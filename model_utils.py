import tensorflow as tf
import tensorflow.keras.layers as layers

bn_mom = 0.1


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
        x = tf.image.resize(x, size=(height2, width2), method='bilinear')

    return x


# Deep Aggregation Pyramid Pooling Module
def DAPPPM(x_in, branch_planes, outplanes):
    input_shape = tf.keras.backend.int_shape(x_in)
    height = input_shape[1]
    width = input_shape[2]
    # Average pooling kernel size
    kernal_sizes_height = [5, 9, 17, height]
    kernal_sizes_width = [5, 9, 17, width]
    # Average pooling strides size
    stride_sizes_height = [2, 4, 8, height]
    stride_sizes_width = [2, 4, 8, width]
    x_list = []

    # y1
    scale0 = layers.BatchNormalization(momentum=bn_mom)(x_in)
    scale0 = layers.Activation("relu")(scale0)
    scale0 = layers.Conv2D(branch_planes, kernel_size=(1, 1), use_bias=False, )(scale0)
    x_list.append(scale0)

    for i in range(len(kernal_sizes_height)):
        # first apply average pooling
        temp = layers.AveragePooling2D(pool_size=(kernal_sizes_height[i], kernal_sizes_width[i]),
                                       strides=(stride_sizes_height[i], stride_sizes_width[i]),
                                       padding="same")(x_in)
        temp = layers.BatchNormalization(momentum=bn_mom)(temp)
        temp = layers.Activation("relu")(temp)
        # then apply 1*1 conv
        temp = layers.Conv2D(branch_planes, kernel_size=(1, 1), use_bias=False, )(temp)
        # then resize using bilinear
        temp = tf.image.resize(temp, size=(height, width), )
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
    kernal_sizes_width = [5, 9, 17, width]
    # Average pooling strides size
    stride_sizes_height = [2, 4, 8, height]
    stride_sizes_width = [2, 4, 8, width]
    x_list = []

    scale0 = layers.BatchNormalization(momentum=bn_mom)(x_in)
    scale0 = layers.Activation("relu")(scale0)
    scale0 = layers.Conv2D(branch_planes, kernel_size=(1, 1), use_bias=False, )(scale0)

    for i in range(len(kernal_sizes_height)):
        # first apply average pooling
        temp = layers.AveragePooling2D(pool_size=(kernal_sizes_height[i], kernal_sizes_width[i]),
                                       strides=(stride_sizes_height[i], stride_sizes_width[i]),
                                       padding="same")(x_in)
        temp = layers.BatchNormalization(momentum=bn_mom)(temp)
        temp = layers.Activation("relu")(temp)
        # then apply 1*1 conv
        temp = layers.Conv2D(branch_planes, kernel_size=(1, 1), use_bias=False, )(temp)
        # then resize using bilinear
        temp = tf.image.resize(temp, size=(height, width), method='bilinear')
        temp = layers.Add()([temp, scale0])

        x_list.append(temp)

    # concatenate all
    combined = layers.concatenate(x_list, axis=-1)

    # scale_out
    combined = layers.BatchNormalization(momentum=bn_mom)(combined)
    combined = layers.Activation("relu")(combined)
    combined = layers.Conv2D(branch_planes * 4, kernel_size=(3, 3), use_bias=False, padding="same", groups=4)(combined)

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
