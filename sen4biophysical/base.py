import tensorflow as tf
import numpy as np

_MINMAX = [
    [0,0,0,0,0.00397272701894,0.0166901380821,0.00638807419226,0],
    [0.263061520472,0.300393577911,0.315398915249,0.618900395798,0.763827384323,0.792011770669,0.503761397883,0.50302598446]
]

def tansig(x):
    return (2.0 / (1.0 + tf.math.exp(-2.0 * x))) - 1.0

class Normalize(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.minmax = self.add_weight(name='minmax', 
                                     shape=(2, 11),
                                     initializer=tf.keras.initializers.zeros(),
                                     trainable=False)

        super(Normalize, self).build(input_shape)

    def call(self, x):
        return 2 * (x - self.minmax[0]) / (self.minmax[1] - self.minmax[0]) - 1

    def compute_output_shape(self, input_shape):
        return input_shape

class Denormalize(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Denormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.minmax = self.add_weight(name='minmax', 
                                        shape=(2, 1),
                                        initializer=tf.keras.initializers.zeros(),
                                        trainable=False)

        super(Denormalize, self).build(input_shape) 

    def call(self, x):
        return 0.5 * (x + 1) * (self.minmax[1] - self.minmax[0]) + self.minmax[0]

    def compute_output_shape(self, input_shape):
        return input_shape
        

class Biophysical(tf.keras.Model):
    def __init__(self, extreme=None, **kwargs):
        super(Biophysical, self).__init__(**kwargs)
        self.extreme = np.array(extreme)
        self.minmax = np.array(_MINMAX)
        
        self.norm = Normalize()
        self.d1 = tf.keras.layers.Dense(5, input_shape=(None, 11))
        self.act = tf.keras.layers.Activation(tansig)
        self.d2 = tf.keras.layers.Dense(1)
        self.denorm = Denormalize()

    def call(self, image, training=False):
        if training is False:
            input_out_of_range = self._validate_input(image)

        x = self.norm(image)
        x = self.d1(x)
        x = self.act(x)
        x = self.d2(x)
        x = self.denorm(x)
        
        if training is False:
            x, mask = self._validate_output(x, input_out_of_range)
            return x, mask
        else:
            return x
    
    def _validate_input(self, x):
        x = x.numpy()
        mask = np.logical_and(np.any(x[..., 0:8] < self.minmax[0], axis=-1),
                              np.any(x[..., 0:8] > self.minmax[1], axis=-1))
        return np.expand_dims(mask, axis=-1)
    
    def _validate_output(self, x, input_out_of_range):
        x = x.numpy()
        output_min = x < self.extreme[1] + self.extreme[0]
        output_max = x > self.extreme[2] + -1*self.extreme[0]
        
        output_min_t = np.logical_and(x < self.extreme[1], x > self.extreme[1] + self.extreme[0])
        output_max_t = np.logical_and(x > self.extreme[2], x < self.extreme[2] + -1*self.extreme[0])
        
        x[output_min_t] = self.extreme[1]
        x[output_max_t] = self.extreme[2]
    
        mask = input_out_of_range + output_min_t + output_max_t + output_min + output_max
        
        return x, mask