# Following https://raw.githubusercontent.com/tensorflow/tensorflow/v2.4.0/tensorflow/python/keras/applications/densenet.py
import tensorflow as tf
from tensorflow.keras import layers, backend
from tensorflow.keras.models import Model


EPSILON = 1.e-3
CONV1_CHANNELS = 8
GROWTH_RATE = 16


def dense_block(x, blocks, name):
    """A dense block.

    Arguments:
      x: input tensor.
      blocks: integer, the number of building blocks.
      name: string, block label.

    Returns:
      Output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, GROWTH_RATE, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.

    Arguments:
      x: input tensor.
      reduction: float, compression rate at transition layers.
      name: string, block label.

    Returns:
      output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=EPSILON, name=name + '_bn')(
            x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(
        int(backend.int_shape(x)[bn_axis] * reduction),
        1,
        use_bias=False,
        name=name + '_conv')(
            x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    Arguments:
      x: input tensor.
      growth_rate: float, growth rate at dense layers.
      name: string, block label.

    Returns:
      Output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x1 = layers.BatchNormalization(
        axis=bn_axis, epsilon=EPSILON, name=name + '_0_bn')(
            x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(
        4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(
            x1)
    x1 = layers.BatchNormalization(
        axis=bn_axis, epsilon=EPSILON, name=name + '_1_bn')(
            x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(
        growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(
            x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet(
        blocks,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax'):
    """Instantiates the DenseNet architecture.

    Reference:
    - [Densely Connected Convolutional Networks](
        https://arxiv.org/abs/1608.06993) (CVPR 2017)

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Note: each Keras Application expects a specific kind of input preprocessing.
    For DenseNet, call `tf.keras.applications.densenet.preprocess_input` on your
    inputs before passing them to the model.

    Arguments:
      blocks: numbers of building blocks for the four dense layers.
      include_top: whether to include the fully-connected
        layer at the top of the network.
      weights: one of `None` (random initialization),
        'imagenet' (pre-training on ImageNet),
        or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
        if `include_top` is False (otherwise the input shape
        has to be `(224, 224, 3)` (with `'channels_last'` data format)
        or `(3, 224, 224)` (with `'channels_first'` data format).
        It should have exactly 3 inputs channels,
        and width and height should be no smaller than 32.
        E.g. `(200, 200, 3)` would be one valid value.
      pooling: optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional block.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional block, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
      classes: optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
      A `keras.Model` instance.

    Raises:
      ValueError: in case of invalid argument for `weights`,
        or invalid input shape.
      ValueError: if `classifier_activation` is not `softmax` or `None` when
        using a pretrained top layer.
    """

    img_input = layers.Input(shape=input_shape)

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(CONV1_CHANNELS, 5, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=EPSILON, name='conv1/bn')(
            x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    for i, block in enumerate(blocks[:-1]):
        x = dense_block(x, block, name=f'conv{i + 2}')
        x = transition_block(x, 0.5, name=f'pool{i + 2}')
    x = dense_block(x, blocks[-1], name=f'conv{i + 3}')

    x = layers.BatchNormalization(axis=bn_axis, epsilon=EPSILON, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation=classifier_activation,
                         name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    model = Model(img_input, x, name='densenet')
    return model


def densenet_custom():
    model = DenseNet([3, 6, 12], weights=None, input_shape=(64, 64, 3), classes=200)
    return model


if __name__ == "__main__":
    model = densenet_custom()
    model.summary()