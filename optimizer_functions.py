from tensorflow.python.keras.api._v2.keras import optimizers
import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow.keras.backend as backend

# https://github.com/tensorflow/addons/issues/844#issuecomment-662087999
class SerializableSGDW(tfa.optimizers.SGDW):
    def get_config(self):
        config = tf.keras.optimizers.SGD.get_config(self)

        config.update(
            {"weight_decay": self._fixed_serialize_hyperparameter("weight_decay"),}
        )

        return config

    def _fixed_serialize_hyperparameter(self, hyperparameter_name):
        """Serialize a hyperparameter that can be a float, callable, or Tensor."""
        value = self._hyper[hyperparameter_name]

        # First resolve the callable
        if callable(value):
            value = value()

        if isinstance(value, tf.keras.optimizers.schedules.LearningRateSchedule):
            return tf.keras.optimizers.schedules.serialize(value)

        if tf.is_tensor(value):
            return backend.get_value(value)

        return value


def sgdw_triangle(lr_init, lr_max, lr_cycle_length, 
    wd_init, wd_decay_per_cycle):
    # https://github.com/tensorflow/addons/issues/844#issuecomment-590903015
    # TODO: Check if it indeed does lR and WD scheduling 

    lr_schedule = tfa.optimizers.TriangularCyclicalLearningRate(lr_init, lr_max,
        step_size=lr_cycle_length//2)

    # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
    wd_schedule = tf.optimizers.schedules.ExponentialDecay(wd_init, 
        lr_cycle_length, wd_decay_per_cycle, staircase=True)

    optimizer = SerializableSGDW(learning_rate=lr_schedule, weight_decay=wd_init,
        momentum=0.9, nesterov=True)
    optimizer.weight_decay = lambda : wd_schedule(optimizer.iterations)
    return optimizer