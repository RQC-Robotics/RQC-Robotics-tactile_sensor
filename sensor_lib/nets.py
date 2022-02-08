import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential

class SensorNN3(Model):
    def __init__(self, input_shape, output_shape):
        super(SensorNN3, self).__init__()
        self.sequential = tf.keras.Sequential([layers.Conv1D(100, 5, strides=2, activation='relu', kernel_initializer='random_normal'),
                                               layers.Conv1D(200, 5, strides=2, activation='relu', kernel_initializer='random_normal'),
                                               layers.Conv1D(400, 5, strides=3, activation='relu', kernel_initializer='random_normal'),
                                               layers.Reshape([400*3]),
                                               layers.Dense(900, activation='relu', kernel_initializer='random_normal'),
                                               layers.Dense(30*30, activation='relu'),
                                               layers.Dense(64*64),
                                               layers.Reshape(output_shape)])
    def call(self, x):
        return self.sequential(x) 
    
class SensorNN3B(Model):
    def __init__(self, input_shape, output_shape):
        super(SensorNN3B, self).__init__()
        self.sequential = tf.keras.Sequential([layers.Conv1D(50, 5, strides=2, activation='relu', kernel_initializer='random_normal'),
                                               layers.Conv1D(200, 5, strides=2, activation='relu', kernel_initializer='random_normal'),
                                               layers.Conv1D(800, 5, strides=3, activation='relu', kernel_initializer='random_normal'),
                                               layers.Reshape([800*3]),
                                               layers.Dense(1800, activation='relu', kernel_initializer='random_normal'),
                                               layers.Dense(60*60, activation='relu'),
                                               layers.Dense(64*64),
                                               layers.Reshape(output_shape)])
    def call(self, x):
        return self.sequential(x) 

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
        super(SensorNN5S, self).__init__()
        self.sequential = tf.keras.Sequential([layers.Reshape((input_shape[0], input_shape[1], 1)),
                                               layers.Conv2D(4, (5, 1), strides=(1, 1), activation='relu', kernel_initializer='random_normal'),
                                               layers.Conv2D(16, (5, 1), strides=(1, 1), activation='relu', kernel_initializer='random_normal'),
                                               layers.Conv2D(8*16, (5, input_shape[1]), activation='relu', kernel_initializer='random_normal'),
                                               layers.Conv2D(8*16, (5, 1), activation='relu', kernel_initializer='random_normal'),
                                               layers.Flatten(),
                                               layers.Dense(300, activation='relu', kernel_initializer='random_normal'),
                                               layers.Dense(30*30, activation='relu'),
                                               layers.Dense(64*64),
                                               layers.Reshape(output_shape)])
    def call(self, x):
        return self.sequential(x)
    
class SensorNN5S_norm(Model):
    def __init__(self, input_shape, output_shape):
        super(SensorNN5S_norm, self).__init__()
        self.sequential = tf.keras.Sequential([layers.Reshape((input_shape[0], input_shape[1], 1)),
                                               layers.Normalization(axis=None),
                                               layers.Conv2D(4, (5, 1), strides=(1, 1), activation='relu', kernel_initializer='random_normal'),
                                               layers.Normalization(axis=None),
                                               layers.Conv2D(16, (5, 1), strides=(1, 1), activation='relu', kernel_initializer='random_normal'),
                                               layers.Normalization(axis=None),
                                               layers.Conv2D(8*16, (5, input_shape[1]), activation='relu', kernel_initializer='random_normal'),
                                               layers.Normalization(axis=None),
                                               layers.Conv2D(8*16, (5, 1), activation='relu', kernel_initializer='random_normal'),
                                               layers.Flatten(),
                                               layers.Dense(300, activation='relu', kernel_initializer='random_normal'),
                                               layers.Dense(30*30, activation='relu'),
                                               layers.Dense(64*64),
                                               layers.Reshape(output_shape)])
    def call(self, x):
        return self.sequential(x)
    
class SensorNN5S_norm_mini(Model):
    def __init__(self, input_shape, output_shape):
        super(SensorNN5S_norm_mini, self).__init__()
        self.sequential = tf.keras.Sequential([layers.Reshape((input_shape[0], input_shape[1], 1)),
                                               layers.Normalization(axis=None),
                                               layers.Conv2D(4, (5, 1), strides=(1, 1), activation='relu', kernel_initializer='random_normal'),
                                               layers.Normalization(axis=None),
                                               layers.Conv2D(16, (5, 1), strides=(1, 1), activation='relu', kernel_initializer='random_normal'),
                                               layers.Normalization(axis=None),
                                               layers.Conv2D(4*16, (5, input_shape[1]), activation='relu', kernel_initializer='random_normal'),
                                               layers.Normalization(axis=None),
                                               layers.Flatten(),
                                               layers.Dense(300, activation='relu', kernel_initializer='random_normal'),
                                               layers.Dense(30*30, activation='relu'),
                                               layers.Dense(64*64),
                                               layers.Reshape(output_shape)])
    def call(self, x):
        return self.sequential(x)

class SensorNN(Model):
    def __init__(self, input_shape, output_shape):
        super(SensorNN, self).__init__()
        self.sequential = tf.keras.Sequential([layers.Flatten(input_shape=input_shape),
                                               layers.Dense(600, activation='relu'),
                                               layers.Dense(600, activation='relu'),
                                               layers.Dense(600, activation='relu'),
                                               layers.Dense(output_shape[0]*output_shape[1], activation='relu'),
                                               layers.Reshape(output_shape)])
    def call(self, x):
        return self.sequential(x)
