import tensorflow as tf
import sensor_lib.data_analis as ds
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
def generate_gaus_params(x, y,size_kof):
    theta = np.pi*np.random.random()
    a = random.lognormvariate(0, 0.8)/(x+y)*size_kof
    b = random.lognormvariate(0, 0.8)/(x+y)*size_kof

    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    M = np.array(((a, 0), (0, b)))
    # while True:
    #   P = np.random.rand(1,2)*np.array([x, y])/3
    #   if np.linalg.norm(P) <= x/3:
    #     P = P + np.array([x, y])/3
    #     break
    P = np.random.rand(1,2)*np.array([x, y])/2 + np.array([x, y])/4
    return np.dot(np.dot(R, M), R.transpose()), P

@jit(nopython=True)
def gaussian_func(X, Y, M, mu, vec_mat):
    vec_mat=get_vec_mat(X,Y)
    x = vec_mat - mu.transpose()
    f=np.zeros((X,Y,1))
    for i in range(X):
        for j in range(Y):
            r=np.dot(M,x[i,j,:,:])
            r2=np.dot(x[i,j,:,:].transpose(),r)
            f[i,j,:]=r2
    # print(f.shape)
    return np.exp(f*-1)[:,:,0]

@jit(nopython=True)
def generate_multi_gaussian(x, y, n,vec_mat,size_kof):
    mat = np.zeros((x, y), dtype=np.float32)
    for i in range(n):
        M, mu = generate_gaus_params(float(x),float(y),size_kof)
        gauss_mat = gaussian_func(x,y,M, mu, vec_mat)
        mat += gauss_mat*random.random()
    return mat

@jit(parallel = True)
def generate_multi_gaussian_alot(x,y,n_images, n,dd):
    x=int(x)
    y=int(y)
    n_images=int(n_images)
    n=int(n)
    vec_mat=get_vec_mat(x,y)
    pressure_mat=np.zeros((n_images,x,y),dtype=np.float32)
    for i in range(n_images):
        pressure_mat[i,:,:]=generate_multi_gaussian(x, y, n, vec_mat, dd)
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

def generate_pressure_map(n_images, n_gauses , x= 97, y = 97, part='fresh_gauss.npy',size_g=6):
  hat_mat=round_fun((x, y), (int(x/2), int(x/2)), lambda r: hat(r,int(x/3)))
  pressure_mat = generate_multi_gaussian_alot(x, y, n_images, n_gauses, size_g)
  pressure_mat_a = pressure_mat*hat_mat
  pressure_mat_a=pressure_mat_a.astype('float32') 
  with open(part, 'wb') as f:
    np.save(f, pressure_mat_a)

def loss_fun(input,alf):
  return tf.keras.layers.Activation(
      lambda x: tf.math.sin(
          alf * tf.math.minimum(tf.math.square(x),tf.math.square(np.pi/2))   
           ))(input)

def sum_losses(input):
  trans = 1-input
  return 1 - tf.reduce_prod(trans, axis=1, keepdims=False)   
 
def visual_for_test(ten,fun='img'):
  mas = ten.numpy()
  n_ang = mas.shape[-1]
  l = mas.shape[0]
  l2=len(mas.shape)
  angls=np.linspace(0, 180, n_ang, endpoint=False)
  dic={}
  ldic={}
  for i in range(n_ang):
    dic2={}
    ldic2={}
    for j in range(l):
      if l2==4: dic2[j] = mas[j,:,:,i]    
      else: dic2[j] = mas[j,:,i]
      ldic2[j] = np.amax(mas)
    dic[(0,angls[i])] = dic2
    ldic[(0,angls[i])] = ldic2
  if fun=='img': fun = lambda x,mas: x.imshow(mas)
  if fun=='plt': fun = lambda x,mas: x.plot(mas)
  ds.show_gerd(dic,fun,ldic)    
    
def fiber_sim(pressure_mat, n_angles, fwhm=20, m=None):
    n_images = pressure_mat.shape[0]
    X = pressure_mat.shape[1]
    Y = pressure_mat.shape[2]

    pressure_mat = tf.constant(pressure_mat,dtype=tf.float32)
    pressure_mat = pressure_mat[:, :, :, tf.newaxis]
    if m==None:
      pressure_mat_angl=pressure_mat
      m=1
    else:
      pressure_mat2=tf.tile(pressure_mat,[m,1,1,1])
      pressure_mat_angl=tf.keras.layers.RandomRotation(
                          (0, 2*np.pi), fill_mode='constant', interpolation='bilinear',
                          seed=None, fill_value=0.0)(pressure_mat2)
    pressure_tensor = pressure_mat_angl[:,:,:,0]
    # pressure_tensor = tf.tile(pressure_tensor,[m_std,1,1])
    # pressure_mat_angl_nose = add_nose(pressure_mat_angl,std,m_std)
    rotated_array = []
    for i in range (n_angles):
      rot_mat = rotate(pressure_mat_angl, i/n_angles/2)
      rotated_array.append(rot_mat)
    rot_tensor = tf.concat(rotated_array, axis=-1)
    
    pressure_tensor = tf.slice(pressure_tensor, [0, int(X/6.0), int(Y/6.0)], [n_images*m, int(X*(1.0 - 2.0/6.0)), int(Y*(1.0 - 2.0/6.0))])
    sliced_tensor = tf.slice(rot_tensor, [0, int(X/6.0), int(Y/6.0), 0], [n_images*m, int(X*(1.0 - 2.0/6.0)), int(Y*(1.0 - 2.0/6.0)), n_angles])
    blured_mat = gauss_blur(sliced_tensor, n_angles, kern_size=50, fwhm=fwhm)
    sq_deriv_tensor =  square(derivate(blured_mat,n_angles))
    sum_tensor = summ(sq_deriv_tensor)

    return sum_tensor, pressure_tensor

def fiber_real_sim(pressure_mat, cfg):
    n_angles = cfg['num different fibers directions']
    m = cfg['num random rotation']
    if m=='None': m=None
    x = cfg['distanse between fibers']
    fwhm = cfg['width of fiber sensibility']['value']
    kernl_size = cfg['width of fiber sensibility']['accuracy']
    fwhm = fwhm/x
    kernl_size =min(int(kernl_size*fwhm),64)
    alf = cfg['kof between aply forse and loss']
    test=cfg['test mod']

    n_images = pressure_mat.shape[0]
    X = pressure_mat.shape[1]
    Y = pressure_mat.shape[2]

    pressure_mat = tf.constant(pressure_mat,dtype=tf.float32)
    pressure_mat = pressure_mat[:, :, :, tf.newaxis]
    if m==None:
      pressure_mat_angl=pressure_mat
      m=1
    else:
      pressure_mat2=tf.tile(pressure_mat,[m,1,1,1])
      pressure_mat_angl=tf.keras.layers.RandomRotation(
                          (0, 1), fill_mode='constant', interpolation='bilinear',
                          seed=None, fill_value=0.0)(pressure_mat2)
    pressure_tensor = pressure_mat_angl[:,:,:,0]
    rotated_array = []
    for i in range (n_angles):
      rot_mat = rotate(pressure_mat_angl, i/n_angles/2)
      rotated_array.append(rot_mat)
    rot_tensor = tf.concat(rotated_array, axis=-1)
    if test:
      print('after_fiber_rot')
      visual_for_test(rot_tensor)
    pressure_tensor = tf.slice(pressure_tensor, [0, int(X/6.0), int(Y/6.0)], [n_images*m, int(X*(1.0 - 2.0/6.0)), int(Y*(1.0 - 2.0/6.0))])
    sliced_tensor = tf.slice(rot_tensor, [0, int(X/6.0), int(Y/6.0), 0], [n_images*m, int(X*(1.0 - 2.0/6.0)), int(Y*(1.0 - 2.0/6.0)), n_angles])
    if test:
      print('after_slise')
      visual_for_test(sliced_tensor)
    blured_mat = gauss_blur(sliced_tensor, n_angles, kern_size=kernl_size, fwhm=fwhm)
    if test:  
      print('after_blur')
      visual_for_test(blured_mat)
    print(alf)
    sq_deriv_tensor =  loss_fun(derivate(blured_mat, n_angles),alf)
    if test:
      print('loss_fun')
      visual_for_test(sq_deriv_tensor)
    sum_tensor = sum_losses(sq_deriv_tensor)
    if test:
      print('sum_loss')
      visual_for_test(sum_tensor,fun='plt')
    std=cfg['reletive nose in fiber transmition detection']
    delt=cfg['nose in fiber transmition detection']
    signal=tf.random.normal((n_images,64,n_angles),mean=1,stddev=std)*sum_tensor + tf.random.normal((n_images,64,n_angles),mean=0,stddev=delt)
    if test:
      print('signal')
      visual_for_test(signal,fun='plt')
    return signal, pressure_tensor

def sim_on_gpu(part, n_random_rot=None, n_angles=4, fwhm=10, batch_size_preproc=128,size=None,test_size=None,max_possible_size=70000,n_del=1):
  with open(part, 'rb') as f: # /content/drive/MyDrive/Colab_projects/fresh_gauss.npy
    mas = np.load(f)
  if size == None:
    size=min(mas.shape[0],max_possible_size)
  if test_size == None:
    test_size = int(size/10) 
  mas=mas[0:size] 
  mas=mas.astype('float32') 
  dataset = tf.data.Dataset.from_tensor_slices(mas[0:-test_size])
  batches = dataset.batch(batch_size_preproc, drop_remainder=False)
  dataset_test = tf.data.Dataset.from_tensor_slices(mas[-test_size:])
  batches_test = dataset_test.batch(batch_size_preproc, drop_remainder=False)
  # batches.map(lambda img: generate_dataset_gpu2(16, 4, tf.constant(img,dtype=tf.float32)))
  input=[]
  output=[]
  for batch in batches:
    input1, output1 = fiber_sim(batch, n_angles, fwhm, n_random_rot)
    input1=input1[:,::n_del,:,:]
    input1=tf.tile(input1,[1,n_del,1,1])
    input.append(input1)
    output.append(output1)
  input=np.concatenate(input)
  output=np.concatenate(output)
  input=input[:,:,0,:]

  input_test=[]
  output_test=[]
  for batch in batches_test:
    input_test1, output_test1 = fiber_sim(batch, n_angles)
    input_test1=input_test1[:,::n_del,:,:]
    input_test1=tf.tile(input_test1,[1,n_del,1,1])
    input_test.append(input_test1)
    output_test.append(output_test1)
  input_test=np.concatenate(input_test)
  output_test=np.concatenate(output_test)
  input_test=input_test[:,:,0,:]
  return input, output, input_test, output_test

def sim_on_gpu2(path,bath_conf,sim_conf):
  size = bath_conf['size']
  max_possible_size = bath_conf['max_possible_size']
  test_size = bath_conf['test_size']
  batch_size_preproc = bath_conf['batch_size_preproc']
  n_del = bath_conf['n_del']
  mas = np.load(path, mmap_mode='r')
  if size == None:
    size=min(mas.shape[0],max_possible_size)
  if test_size == None:
    test_size = int(size/10) 
  mas=mas[0:size] 
  mas=mas.astype('float32') 
  dataset = tf.data.Dataset.from_tensor_slices(mas[0:-test_size])
  batches = dataset.batch(batch_size_preproc, drop_remainder=False)
  dataset_test = tf.data.Dataset.from_tensor_slices(mas[-test_size:])
  batches_test = dataset_test.batch(batch_size_preproc, drop_remainder=False)

  input=[]
  output=[]
  for batch in batches:
    input1, output1 = fiber_real_sim(batch, sim_conf)
    input1=input1[:,::n_del,:]
    input1=tf.tile(input1,[1,n_del,1])
    input.append(input1)
    output.append(output1)
  input=np.concatenate(input)
  output=np.concatenate(output)

  input_test=[]
  output_test=[]
  for batch in batches_test:
    input_test1, output_test1 = fiber_real_sim(batch, sim_conf)
    input_test1=input_test1[:,::n_del,:]
    input_test1=tf.tile(input_test1,[1,n_del,1])
    input_test.append(input_test1)
    output_test.append(output_test1)
  input_test=np.concatenate(input_test)
  output_test=np.concatenate(output_test)
  return input, output, input_test, output_test

def prepare_dataset_for_train(input, output, batch_size_fit_model=1024):
  dataset= tf.data.Dataset.from_tensor_slices((input,output))
  dataset_b = dataset.batch(batch_size_fit_model)
  return dataset_b, dataset

