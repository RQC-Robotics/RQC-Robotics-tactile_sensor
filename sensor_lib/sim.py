import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras import regularizers
# from tensorflow.keras.models import Model
# from tensorflow.keras import Sequential
from tensorflow.data import Dataset
# import matplotlib.pyplot as plt
import numpy as np
import random
import math
from numba import jit
# import numba

@jit(nopython=True)
def get_vec_mat(x,y):
    mas=np.zeros((x,y,2,1),dtype=np.float32)
    for i in range(x):
        for j in range(y):
            mas[i,j,:,0]=[i,j]
    return mas   

@jit(nopython=True)
def generate_gaus_params(x, y):
    theta = np.pi*np.random.random()
    a = np.random.random()/(x+y)*8.0
    b = np.random.random()/(x+y)*8.0

    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    M = np.array(((a, 0), (0, b)))
    return np.dot(np.dot(R, M), R.transpose()), np.random.rand(1,2)*np.array([x, y])/3 + np.array([x, y])/3

@jit(nopython=True)
def gaussian_func(x, y, M, mu, vec_mat):
    vec_mat=get_vec_mat(x,y)
    x = vec_mat - mu.transpose()
    f=np.zeros((x,y,1))
    for i in range(x):
        for j in range(y):
            r=np.dot(M,x[i,j,:,:])
            r2=np.dot(x[i,j,:,:].transpose(),r)
            f[i,j,:]=r2
    # print(f.shape)
    return np.exp(f*-1)[:,:,0]

@jit(nopython=True)
def generate_multi_gaussian(x, y, n,vec_mat):
    mat = np.zeros((x, y), dtype=np.float32)
    for i in range(n):
        M, mu = generate_gaus_params(float(x),float(y))
        gauss_mat = gaussian_func(x,y,M, mu, vec_mat)
        mat += gauss_mat*random.random()
    return mat

@jit(parallel = True)
def generate_multi_gaussian_alot(x,y,n_images, n):
    vec_mat=get_vec_mat(X,Y)
    pressure_mat=np.zeros((n_images,x,y),dtype=np.float32)
    for i in range(n_images):
        pressure_mat[i,:,:]=generate_multi_gaussian(x, y, n, vec_mat)
    return pressure_mat

def Convolution(input, filter, padding="SAME"):
  convolved = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding=padding)
  return convolved

def gauss_blur(input, num_angles, kern_size=40, fwhm=20):
    gauss_kernel = makeGaussian(size=kern_size, fwhm=fwhm)
    gauss_kernel = tf.tile(gauss_kernel, [1, 1, 1, num_angles])
    return Convolution(input, gauss_kernel, padding="SAME")

def makeGaussian(size, fwhm = 3, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    gauss = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    #plt.imshow(gauss)
    gauss = tf.constant(gauss, dtype=tf.float32)[:-1, :-1, tf.newaxis, tf.newaxis]
    return gauss

def rotate(input, theta):
    rot_layer = tf.keras.layers.RandomRotation(
      (theta, theta), fill_mode='constant', interpolation='bilinear',
      seed=None, fill_value=0.0)
    return rot_layer(input)

def derivate(input, num_angles):
    derivative_kernel = tf.constant(np.array([[1, -2, 1]]).transpose(), dtype=tf.float32)
    derivative_kernel = derivative_kernel[:, :, tf.newaxis, tf.newaxis]
    derivative_kernel = tf.tile(derivative_kernel, [1, 1, 1, num_angles])
    derivative_mat = Convolution(input, derivative_kernel, padding='VALID')
    return derivative_mat

def square(input):
    return tf.keras.layers.Activation(tf.math.square)(input)

def summ(input):
    return tf.transpose(tf.math.reduce_sum(input, axis=1, keepdims=True),[0,2,1,3])

def add_nose(input,std,n):
  input2=tf.tile(input,[n,1,1,1])
  noise = 1 + tf.random.normal(shape=tf.shape(input2), mean=0.0, stddev=std, dtype=tf.float32)
  return input2*noise

def round_fun(shape,centr,fun):
  center=np.array(centr)
  mas=np.zeros(shape)
  for x in range(shape[0]):
    for y in range(shape[1]):
      p=np.array((x,y))
      mas[x,y]=fun(np.linalg.norm(p-center))
  return mas

def hat(x,r):
  if abs(x)<r:
    resalt=math.exp(1/8/((x/r)**2-1**2))
  else:
    resalt=0
  return resalt

def generate_pressure_map(n_images, x= 97, y = 97, part='fresh_gauss.npy'):
  hat_mat=round_fun((x, y), (int(x/2), int(x/2)), lambda r: hat(r,int(x/3)))
  pressure_mat = generate_multi_gaussian_alot(x,y,n_images)
  pressure_mat_a = pressure_mat*hat_mat
  with open(part, 'wb') as f:
    np.save(f, pressure_mat_a)

def fiber_sim(m, n_angles, pressure_mat):
    n_images = pressure_mat.shape[0]
    X = pressure_mat.shape[1]
    Y = pressure_mat.shape[2]

    pressure_mat = tf.constant(pressure_mat,dtype=tf.float32)
    pressure_mat_angl = pressure_mat[:, :, :, tf.newaxis]
    # pressure_mat2=tf.tile(pressure_mat,[m,1,1,1])
    # pressure_mat_angl=tf.keras.layers.RandomRotation(
    #   (0, 2*np.pi), fill_mode='constant', interpolation='bilinear',
    #   seed=None, fill_value=0.0)(pressure_mat2)
    pressure_tensor = pressure_mat_angl[:,:,:,0]
    # pressure_tensor = tf.tile(pressure_tensor,[m_std,1,1])
    # pressure_mat_angl_nose = add_nose(pressure_mat_angl,std,m_std)
    rotated_array = []
    for i in range (n_angles):
      rot_mat = rotate(pressure_mat_angl, i*np.pi/n_angles)
      rotated_array.append(rot_mat)
    rot_tensor = tf.concat(rotated_array, axis=-1)
    
    pressure_tensor = tf.slice(pressure_tensor, [0, int(X/6.0), int(Y/6.0)], [n_images*m, int(X*(1.0 - 2.0/6.0)), int(Y*(1.0 - 2.0/6.0))])
    sliced_tensor = tf.slice(rot_tensor, [0, int(X/6.0), int(Y/6.0), 0], [n_images*m, int(X*(1.0 - 2.0/6.0)), int(Y*(1.0 - 2.0/6.0)), n_angles])
    blured_mat = gauss_blur(sliced_tensor, n_angles, kern_size=50, fwhm=20)
    sq_deriv_tensor =  square(derivate(blured_mat,n_angles))
    sum_tensor = summ(sq_deriv_tensor)

    return sum_tensor, pressure_tensor

def sim_on_gpu(part, n_random_rot=16, n_angles=4, batch_size_preproc=128):
  with open(part, 'rb') as f: # /content/drive/MyDrive/Colab_projects/fresh_gauss.npy
    mas = np.load(f)
    mas=mas[0:70000] 
  mas=mas.astype('float32') 
  dataset = tf.data.Dataset.from_tensor_slices(mas[0:-1000])
  batches = dataset.batch(batch_size_preproc, drop_remainder=False)
  dataset_test = tf.data.Dataset.from_tensor_slices(mas[-1000:])
  batches_test = dataset_test.batch(batch_size_preproc, drop_remainder=False)
  # batches.map(lambda img: generate_dataset_gpu2(16, 4, tf.constant(img,dtype=tf.float32)))
  input=[]
  output=[]
  for batch in batches:
    input1, output1 = fiber_sim(n_random_rot, n_angles, batch)
    input.append(input1)
    output.append(output1)
  input=np.concatenate(input)
  output=np.concatenate(output)
  input=input[:,:,0,:]

  input_test=[]
  output_test=[]
  for batch in batches_test:
    input_test1, output_test1 = fiber_sim(n_random_rot, n_angles, batch)
    input_test.append(input_test1)
    output_test.append(output_test1)
  input_test=np.concatenate(input_test)
  output_test=np.concatenate(output_test)
  input_test=input_test[:,:,0,:]
  return input, output, input_test, output_test

def prepare_dataset_for_train(input, output, batch_size_fit_model=1024):
  dataset= tf.data.Dataset.from_tensor_slices((input,output))
  dataset_b = dataset.batch(batch_size_fit_model)
  return dataset_b, dataset

