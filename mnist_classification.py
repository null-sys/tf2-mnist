import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

mnist_,mnist_info = tfds.load(name="mnist",with_info=True,as_supervised=True)


#preprocessing
mnist_train,mnist_test = mnist_['train'],mnist_['test']

validation_size = 0.1 * mnist_info.splits['train'].num_examples

validation_size = tf.cast(validation_size,tf.int64)

test_size = mnist_info.splits['train'].num_examples
test_size = tf.cast(test_size,tf.int64)


def scale(image,label):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image,label

scale_train = mnist_train.map(scale)

test_data = mnist_test.map(scale)


BUFFER_SIZE = 10000

suffled_train_and_validation_data = scale_train.shuffle(BUFFER_SIZE)

validation_data = suffled_train_and_validation_data.take(validation_size)
train_data = suffled_train_and_validation_data.skip(validation_size)

BATCH_SIZE = 100

train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(validation_size)
test_data = test_data.batch(test_size)

validation_inputs , validation_targets = next(iter(validation_data))

