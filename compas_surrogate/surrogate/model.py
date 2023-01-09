"""ML Surrogate Model for COMPAS LnL"""

import tensorflow as tf
from tensorflow.keras import layers


class LnLRegressor(tf.keras.Model):
    def __init__(self, num_inputs, num_outputs):
        # Initialize the necessary components of tf.keras.Model
        super(LnLRegressor, self).__init__()
        # Now we initalize the needed layers - order does not matter.
        self.dense1 = layers.Dense(64, activation=tf.nn.relu)
        self.dense2 = layers.Dense(32, activation=tf.nn.relu)
        self.output_layer = layers.Dense(num_outputs)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)
