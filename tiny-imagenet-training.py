# Inspired from https://github.com/ZohebAbai/Tiny-ImageNet-Challenge

import numpy as np
import glob
import os

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ModelCheckpoint

from mobilenet_v2 import MobileNetV2
from densenet_custom import densenet_custom

from optimizer_functions import sgdw_triangle

# debug
import matplotlib.pyplot as plt


seed = 1947
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 120
MAX_BATCH_SIZE = 128
CYCLE_LENGTH_IN_EPOCHS = 40
MIN_LR = 1e-5
MAX_LR = 1e-2
INIT_WD = 1e-5
WD_DECAY_FOR_CYCLE = 0.5

# input image dimensions
IMAGE_SIZE = 64
NUM_CLASSES = 200
NUM_TRAIN_SAMPLES = 100 * 1000

ROOT_DIR = r'C:\stuff\datasets\tiny-imagenet-200'
CACHE = True
SHUFFLE = True
SHUFFLE_SAMPLES = 10 * 1000

autotune = tf.data.experimental.AUTOTUNE


def get_label_from_path(path):
    basename = os.path.basename(path)
    label = basename.split('_')[0]
    return label


def get_train_files_labels(root, max_batch_size):
    pattern = os.path.join(root, 'train/*/images/*.JPEG')
    paths = glob.glob(pattern)
    if SHUFFLE:
        np.random.shuffle(paths)

    # Drop some files to make the number of samples multiple of
    # some number to ease pre-processing in batches
    paths = paths[:(len(paths)//max_batch_size) * max_batch_size]

    text_labels = [get_label_from_path(a) for a in paths]
    classes = set(text_labels)
    num_classes = len(classes)
    class_indices_map = dict(zip(classes, range(num_classes)))
    numeric_labels = [class_indices_map[a] for a in text_labels]

    return paths, numeric_labels, class_indices_map


def get_tf_dataset(paths, labels):
    paths_ds = tf.data.Dataset.from_tensor_slices(paths)
    images_ds = paths_ds.map(lambda x: tf.io.read_file(x), autotune)

    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    datagen = tf.data.Dataset.zip((images_ds, labels_ds))
    return datagen


def get_val_paths_labels(root, class_indices_map):
    annot_file = os.path.join(root, 'val/val_annotations.txt')
    with open(annot_file) as f:
        annots = f.readlines()
    path_label_pairs = [a.split()[:2] for a in annots]
    paths, text_labels = zip(*path_label_pairs)
    paths = [os.path.join(root, 'val/images', a) for a in paths]
    numeric_labels = [class_indices_map[a] for a in text_labels]
    return paths, numeric_labels


def decode_image(x, y):
    # Decode a JPEG-encoded image to a uint8 tensor.
    x = tf.image.decode_jpeg(x, channels=3)
    # Return image ranging from [-1, 1]
    x = tf.cast(x, tf.float32)
    x = (x / 128.) - 1.
    return x, y


# some inspiration from https://www.tensorflow.org/tutorials/images/data_augmentation
def augment(image_size, batch_size):
    def augment_helper(x, y):
        # https://www.tensorflow.org/api_docs/python/tf/image
        # x = tf.image.random_jpeg_quality(x, 80, 95) <-- what should be x range?
        x = tf.image.random_flip_left_right(x)
        x = tf.image.resize_with_crop_or_pad(x, image_size + 16, image_size + 16)
        x = tf.image.random_crop(x, [batch_size, image_size, image_size, 3])
        x = tf.image.random_brightness(x, max_delta=0.1)
        x = tf.image.random_contrast(x, lower=0.8, upper=1.2)
        x = tfa.image.random_cutout(x, mask_size=(4*2))
        noise = tf.random.normal(shape=tf.shape(x), mean=0, stddev=0.03, dtype=tf.float32)
        x = tf.add(x, noise)
        x = tf.clip_by_value(x, -1, 1)
        return x, y
    return augment_helper


def debug_visualize(dataset, num_batches=1):
    for xx, yy in dataset.take(num_batches):
        for x, y in zip(xx, yy):
            im_from_ds = x.numpy()
            im = (im_from_ds + 1)/2
            plt.imshow(im)
            plt.title(f'{np.min(im_from_ds), np.max(im_from_ds)}')
            plt.show()


def get_train_val_datasets():
    # get the train dataset
    paths, labels, class_indices_map = get_train_files_labels(ROOT_DIR, MAX_BATCH_SIZE)
    train_ds = get_tf_dataset(paths, labels)

    # get the val dataset
    paths, labels = get_val_paths_labels(ROOT_DIR, class_indices_map)
    val_ds = get_tf_dataset(paths, labels)

    if CACHE:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    if SHUFFLE:
        train_ds = train_ds.shuffle(SHUFFLE_SAMPLES)

    train_ds = train_ds.map(decode_image, autotune)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.map(augment(IMAGE_SIZE, BATCH_SIZE), autotune)

    val_ds = val_ds.map(decode_image, autotune)
    val_ds = val_ds.batch(BATCH_SIZE)

    return train_ds, val_ds


def get_optimizer():
    # optimizer = tf.optimizers.SGD((MIN_LR + MAX_LR)/2, momentum=0.9, nesterov=True)
    steps_per_epoch = NUM_TRAIN_SAMPLES // BATCH_SIZE
    num_iters_in_cycle = CYCLE_LENGTH_IN_EPOCHS * steps_per_epoch

    # TODO: Haven't yet verified
    optimizer = sgdw_triangle(MIN_LR, MAX_LR, num_iters_in_cycle,
        INIT_WD, WD_DECAY_FOR_CYCLE)
    return optimizer


train_ds, val_ds = get_train_val_datasets()

# debug_visualize(train_ds)

optimizer = get_optimizer()
# model = MobileNetV2((64, 64, 3), weights=None, classes=NUM_CLASSES)
model = densenet_custom()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Callbacks
checkpointer = ModelCheckpoint(filepath=model.name + \
    '-{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}-.h5',
    verbose=1, save_best_only=True, monitor="val_accuracy")


model.fit(x=train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[checkpointer])

print('Finished!')
