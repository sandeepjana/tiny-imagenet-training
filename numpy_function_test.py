import tensorflow as tf
import numpy as np



def my_numpy_fn(x, y):
  return -x, -y

features = np.arange(10).astype(np.float32)
labels = 2 * features
ds = tf.data.Dataset.from_tensor_slices((features, labels))
ds = ds.batch(2)
# Do a transformation that loses rank information.
ds = ds.map(
    lambda x, y: tf.numpy_function(
        my_numpy_fn, inp=[x, y], Tout=[tf.float32, tf.float32]
        ), tf.data.experimental.AUTOTUNE
    )

assert iter(ds).output_shapes[0] == tf.TensorShape(None)

# Model works with Tensors of unknown rank.
# Note that if your Model uses layers like `Dense`, etc. that
# only work with ceratin ranks, you should still use `x.set_shape`
# before passing the data to that layer, to give the Model a hint
# about the rank.
class MyModel(tf.keras.Model):
  def call(self, x):
    return 2 * x

model = MyModel()
model.compile('sgd', 'mse')
model.fit(ds)