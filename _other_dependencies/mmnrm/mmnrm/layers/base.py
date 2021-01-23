import tensorflow as tf
from tensorflow.keras import backend as K


class TrainableLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 initializer='glorot_uniform',
                 regularizer=None,
                 **kwargs):
        super(TrainableLayer, self).__init__(**kwargs)
        self.initializer = initializer
        self.regularizer = regularizer