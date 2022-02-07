import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential

class SensorNN4S(Model):
    def __init__(self, input_shape, output_shape):
        super(SensorNN4S, self).__init__()
        self.sequential = tf.keras.Sequential([layers.Reshape((input_shape[0], input_shape[1], 1)),
                                               layers.Conv2D(4, (5, 1), strides=(1, 1), activation='relu', kernel_initializer='random_normal'),
                                               layers.Conv2D(16, (5, 1), strides=(1, 1), activation='relu', kernel_initializer='random_normal'),
                                               layers.Conv2D(64*16, (5, input_shape[1]), activation='relu', kernel_initializer='random_normal'),
                                               layers.Conv2D(64*16, (5, 1), activation='relu', kernel_initializer='random_normal'),
                                               layers.Flatten(),
                                               layers.Dense(900, activation='relu', kernel_initializer='random_normal'),
                                               layers.Dense(30*30, activation='relu'),
                                               layers.Reshape((30, 30, 1)),
                                               layers.Conv2DTranspose(1, (6, 6), (2, 2), kernel_initializer='random_normal'),
                                               layers.Reshape(output_shape)])
    def call(self, x):
        return self.sequential(x)
    
class SensorNN5S(Model):
    def __init__(self, input_shape, output_shape):
        super(SensorNN4S, self).__init__()
        self.sequential = tf.keras.Sequential([layers.Reshape((input_shape[0], input_shape[1], 1)),
                                               layers.Conv2D(4, (5, 1), strides=(1, 1), activation='relu', kernel_initializer='random_normal'),
                                               layers.Conv2D(16, (5, 1), strides=(1, 1), activation='relu', kernel_initializer='random_normal'),
                                               layers.Conv2D(64*16, (5, input_shape[1]), activation='relu', kernel_initializer='random_normal'),
                                               layers.Conv2D(64*16, (5, 1), activation='relu', kernel_initializer='random_normal'),
                                               layers.Flatten(),
                                               layers.Dense(900, activation='relu', kernel_initializer='random_normal'),
                                               layers.Dense(30*30, activation='relu'),
                                               layers.Dense(64*64),
                                               layers.Reshape(output_shape)])
    def call(self, x):
        return self.sequential(x)
