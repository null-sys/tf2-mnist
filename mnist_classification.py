import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

mnist_,mnist_info = tfds.load(name="mnist",with_info=True,as_supervised=True)

