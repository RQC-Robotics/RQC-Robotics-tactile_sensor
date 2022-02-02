import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from numba import jit
import numba


with open('gausses.npy', 'rb') as f:
    inputt = np.load(f)
    output = np.load(f)
    # sliced_tensor = np.load(f)
    # sq_deriv_tensor = np.load(f)


with tf.device('/gpu:0'):
    inputt=tf.constant(inputt)
    output=tf.constant(output)
    # sliced_tensor=tf.constant(sliced_tensor,dtype=tf.float32)
    # sq_deriv_tensor=tf.constant(sq_deriv_tensor,dtype=tf.float32)


output.shape


with tf.device('/gpu:2'):
    input2=[inputt[:,::2,i] for i in range(4)]
    output2=output[:,::2,::2]


print(output2.shape)
print(input2[0].shape)


n=244
plt.imshow(output[n])
plt.show()
plt.imshow(sq_deriv_tensor[n,:,:,0])
plt.show()
plt.plot(output[n,:,30])
plt.show()
plt.plot(sq_deriv_tensor[n,:,30,0])
plt.show()


def get_siamese_model(input_shape,n_angl):
    """
        Model architecture
    """
    inputs=[layers.Input(input_shape) for _ in range(n_angl)]
    # inputs=tf.tile(layers.Input(input_shape)[tf.newaxis,:,:,:],[n_angl,1,1,1,])
    # Define the tensors for the two input images
    
    # Convolutional Neural Network
    model = Sequential()
    model.add(layers.Conv1D(4, 5, activation='relu', input_shape=input_shape)) # kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)
    model.add(layers.MaxPooling1D())
    model.add(layers.Conv1D(8, 3, activation='relu', input_shape=input_shape))
    # model.add(layers.MaxPooling1D())
    # model.add(layers.Conv1D(16, 3, activation='relu', input_shape=input_shape))
    # model.add(layers.MaxPooling1D())
    # model.add(layers.Conv1D(32, 3, activation='relu', input_shape=input_shape))
    # model.add(layers.MaxPooling1D())
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    
    # Generate the encodings (feature vectors) for the two images
    encodeds = tf.stack([model(input) for input in inputs], axis=1)
    # encodeds = tf.map_fn(model, inputs)
    # encodeds = tf.transpose(encodeds, perm=[1,0,2,3])
    flat_layer = layers.Flatten()(encodeds)
    dense_layer = layers.Dense(32*4, activation='relu')(flat_layer)
    flat_resalt = layers.Dense(31*31, activation='relu')(dense_layer)
    resalt = layers.Reshape((31, 31))(flat_resalt)
    # print(tf.stack(inputs,axis=3))
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=inputs,outputs=resalt)
    
    # return the model
    return siamese_net


model = get_siamese_model((31,1),4)
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])


with tf.device('/gpu:0'):
    model.fit(input2, output2, epochs = 100, validation_split=0.1, verbose=1)


predictions = model.predict(input2)


N = 901
plt.imshow(predictions[N])
plt.show()
plt.imshow(output.numpy()[N])



