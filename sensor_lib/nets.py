import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential


class SensorNN3(Model):

    def __init__(self, input_shape, output_shape):
        super(SensorNN3, self).__init__()
        self.sequential = tf.keras.Sequential([
            layers.Conv1D(100,
                          5,
                          strides=2,
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Conv1D(200,
                          5,
                          strides=2,
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Conv1D(400,
                          5,
                          strides=3,
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Reshape([400 * 3]),
            layers.Dense(900,
                         activation='relu',
                         kernel_initializer='random_normal'),
            layers.Dense(30 * 30, activation='relu'),
            layers.Dense(64 * 64),
            layers.Reshape(output_shape)
        ])

    def call(self, x):
        return self.sequential(x)


class SensorNN3B(Model):

    def __init__(self, input_shape, output_shape):
        super(SensorNN3B, self).__init__()
        self.sequential = tf.keras.Sequential([
            layers.Conv1D(50,
                          5,
                          strides=2,
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Conv1D(200,
                          5,
                          strides=2,
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Conv1D(800,
                          5,
                          strides=3,
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Reshape([800 * 3]),
            layers.Dense(1800,
                         activation='relu',
                         kernel_initializer='random_normal'),
            layers.Dense(60 * 60, activation='relu'),
            layers.Dense(64 * 64),
            layers.Reshape(output_shape)
        ])

    def call(self, x):
        return self.sequential(x)


class SensorNN4S(Model):

    def __init__(self, input_shape, output_shape):
        super(SensorNN4S, self).__init__()
        self.sequential = tf.keras.Sequential([
            layers.Reshape((input_shape[0], input_shape[1], 1)),
            layers.Conv2D(4, (5, 1),
                          strides=(1, 1),
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Conv2D(16, (5, 1),
                          strides=(1, 1),
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Conv2D(64 * 16, (5, input_shape[1]),
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Conv2D(64 * 16, (5, 1),
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Flatten(),
            layers.Dense(900,
                         activation='relu',
                         kernel_initializer='random_normal'),
            layers.Dense(30 * 30, activation='relu'),
            layers.Reshape((30, 30, 1)),
            layers.Conv2DTranspose(1, (6, 6), (2, 2),
                                   kernel_initializer='random_normal'),
            layers.Reshape(output_shape)
        ])

    def call(self, x):
        return self.sequential(x)


class SensorNN5S(Model):

    def __init__(self, input_shape, output_shape):
        super(SensorNN5S, self).__init__()
        self.sequential = tf.keras.Sequential([
            layers.Reshape((input_shape[0], input_shape[1], 1)),
            layers.Conv2D(4, (5, 1),
                          strides=(1, 1),
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Conv2D(16, (5, 1),
                          strides=(1, 1),
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Conv2D(8 * 16, (5, input_shape[1]),
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Conv2D(8 * 16, (5, 1),
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Flatten(),
            layers.Dense(300,
                         activation='relu',
                         kernel_initializer='random_normal'),
            layers.Dense(30 * 30, activation='relu'),
            layers.Dense(64 * 64),
            layers.Reshape(output_shape)
        ])

    def call(self, x):
        return self.sequential(x)


class SensorNN5S_norm(Model):

    def __init__(self, input_shape, output_shape):
        super(SensorNN5S_norm, self).__init__()
        self.sequential = tf.keras.Sequential([
            layers.Reshape((input_shape[0], input_shape[1], 1)),
            layers.Normalization(axis=None),
            layers.Conv2D(4, (5, 1),
                          strides=(1, 1),
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Normalization(axis=None),
            layers.Conv2D(16, (5, 1),
                          strides=(1, 1),
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Normalization(axis=None),
            layers.Conv2D(8 * 16, (5, input_shape[1]),
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Normalization(axis=None),
            layers.Conv2D(8 * 16, (5, 1),
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Flatten(),
            layers.Dense(300,
                         activation='relu',
                         kernel_initializer='random_normal'),
            layers.Dense(30 * 30, activation='relu'),
            layers.Dense(64 * 64),
            layers.Reshape(output_shape)
        ])

    def call(self, x):
        return self.sequential(x)


class SensorNN5S_norm2(Model):

    def __init__(self, input_shape, output_shape):
        super(SensorNN5S_norm2, self).__init__()
        self.sequential = tf.keras.Sequential([
            layers.Reshape((input_shape[0], input_shape[1], 1)),
            layers.Conv2D(4, (5, 1),
                          strides=(1, 1),
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Normalization(axis=None),
            layers.Conv2D(16, (5, 1),
                          strides=(1, 1),
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Conv2D(8 * 16, (5, input_shape[1]),
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Conv2D(16 * 16, (5, input_shape[1]),
                          activation='relu',
                          kernel_initializer='random_normal'),
            layers.Flatten(),
            layers.Dense(900,
                         activation='relu',
                         kernel_initializer='random_normal'),
            layers.Dense(30 * 30, activation='relu'),
            layers.Dense(64 * 64),
            layers.Reshape(output_shape)
        ])

    def call(self, x):
        return self.sequential(x)


class SensorNN(Model):

    def __init__(self, input_shape, output_shape):
        super(SensorNN, self).__init__()
        self.sequential = tf.keras.Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.Dense(600, activation='relu'),
            layers.Dense(600, activation='relu'),
            layers.Dense(600, activation='relu'),
            layers.Dense(output_shape[0] * output_shape[1], activation='relu'),
            layers.Reshape(output_shape)
        ])

    def call(self, x):
        return self.sequential(x)


def SensorNN5S_norm_deep(input_shape, output_shape):
    lay_in = layers.Input(input_shape[1:3])
    lay = layers.Reshape((input_shape[1], input_shape[2], 1))(lay_in)
    lay = layers.Conv2D(8, (3, 1),
                        strides=(1, 1),
                        padding='same',
                        activation='relu',
                        kernel_initializer='random_normal',
                        name='Conv_1.1')(lay)
    lay1 = layers.MaxPool2D((2, 1), strides=(2, 1), name='MaxPool_1.1')(lay)
    lay1 = layers.Normalization(axis=None)(lay1)
    lay1 = layers.Conv2D(64, (3, 1),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         kernel_initializer='random_normal',
                         name='Conv_1.2')(lay1)
    lay1 = layers.MaxPool2D((2, 1), strides=(2, 1), name='MaxPool_1.2')(lay1)
    lay1 = layers.Normalization(axis=None)(lay1)

    lay2 = layers.MaxPool2D((4, 1), strides=(4, 1), name='MaxPool_1.0')(lay)
    lay = layers.concatenate([lay1, lay2], name='concatenate_1')

    lay1 = layers.Conv2D(128, (3, 1),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         kernel_initializer='random_normal',
                         name='Conv_2.1')(lay)
    lay1 = layers.MaxPool2D((2, 1), strides=(2, 1), name='MaxPool_2.1')(lay1)
    lay1 = layers.Normalization(axis=None)(lay1)
    lay1 = layers.Conv2D(128, (3, 1),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         kernel_initializer='random_normal',
                         name='Conv_2.2')(lay1)
    lay1 = layers.MaxPool2D((2, 1), strides=(2, 1), name='MaxPool_2.2')(lay1)

    lay2 = layers.MaxPool2D((4, 1), strides=(4, 1), name='MaxPool_2.0')(lay)
    lay = layers.concatenate([lay1, lay2], name='concatenate_2')

    lay = layers.Conv2D(64, (1, input_shape[2]),
                        strides=(1, 1),
                        activation='relu',
                        kernel_initializer='random_normal',
                        name='Conv_3')(lay)
    lay = layers.Flatten()(lay)
    lay = layers.Dense(15 * 13,
                       activation='relu',
                       name='Dense_4',
                       kernel_initializer='random_normal',
                       bias_initializer='random_normal')(lay)
    lay = layers.Dense(30 * 30,
                       activation='relu',
                       name='Dense_5',
                       kernel_initializer='random_normal',
                       bias_initializer='random_normal')(lay)
    lay = layers.Dense(64 * 64,
                       activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                       name='Dense_6',
                       kernel_initializer='random_normal',
                       bias_initializer='random_normal')(lay)
    lay_out = layers.Reshape(output_shape[1:3])(lay)
    model = tf.keras.Model(lay_in, lay_out)
    return model
