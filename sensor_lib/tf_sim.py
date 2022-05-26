import tensorflow as tf
import sensor_lib.data_analis as ds
from tensorflow.data import Dataset
# import matplotlib.pyplot as plt
import numpy as np
import random
import math
# from numba import jit


def get_vec_mat(x, y):
    mas = np.zeros((x, y, 2), dtype=np.float32)
    for i in range(x):
        for j in range(y):
            mas[i, j, :] = [i, j]
    return mas


def gen_rand_cof(n_gaus, x, y, size_kof, seed):
    np.random.seed(seed)
    theta = np.pi * np.random.random(size=(n_gaus, 1))
    E = np.random.lognormal(0, 0.8, size=(n_gaus, 2)) / (x+y) * size_kof
    P = np.random.rand(n_gaus, 2) * np.array([x, y]) / 2 + np.array([x, y]) / 4
    res = np.concatenate([theta, E, P], axis=1)
    return tf.constant(res, dtype=tf.float32)


# @tf.function(input_signature=(tf.TensorSpec(shape=[5], dtype=tf.float32),))
def generate_gaus_params(data):
    c, s = tf.math.cos(data[0]), tf.math.sin(data[0])
    row1 = tf.stack([c, -s], axis=0)
    row2 = tf.stack([s, c], axis=0)
    R = tf.stack([row1, row2])
    M = tf.linalg.diag(data[1:3])
    form = tf.transpose(R) @ M @ R
    coord = tf.reshape(data[-2:], (1, 2))
    return tf.concat([form, coord], axis=0)


# tf.function
def gaussian_func_flat(g_param, vec_mat):
    M = g_param[:2]
    mu = g_param[2]
    x = vec_mat - mu
    x = tf.reshape(x, [-1, 2, 1])
    f = tf.vectorized_map(lambda r: tf.transpose(r) @ M @ r,
                          x,
                          fallback_to_while_loop=False)
    f = tf.reshape(f, [-1])
    return tf.math.exp(f * -1)


# @tf.function
def generate_multi_gaussian_flat(gaus_data, vec_mat):
    params = tf.vectorized_map(generate_gaus_params, gaus_data)
    gauses = tf.vectorized_map(
        lambda g_param: gaussian_func_flat(g_param, vec_mat), params)
    return tf.reduce_sum(gauses, 0)


# @tf.function
def generate_pictures(gaus_data, vec_mat):
    # vec_mat=tf.constant(get_vec_mat(x,y),dtype=tf.float32)
    # vec_mat=tf.reshape(vec_mat,[-1,2])
    # gaus_data = gen_rand_cof(n_gaus*n_pic,x, y,size_kof)
    # gaus_data = tf.reshape(gaus_data,[n_pic,n_gaus,5])
    # generate_multi_gaussian_flat(gaus_data[0],vec_mat)
    pictures = tf.vectorized_map(
        lambda data: generate_multi_gaussian_flat(data, vec_mat),
        gaus_data,
        fallback_to_while_loop=True)
    # return tf.reshape(pictures,[n_pic,x,y])
    return pictures


def Convolution(input, filter, padding="SAME"):
    convolved = tf.nn.conv2d(input,
                             filter,
                             strides=[1, 1, 1, 1],
                             padding=padding)
    return convolved


def gauss_blur(input, num_angles, kern_size=40, fwhm=20):
    gauss_kernel = makeGaussian(size=kern_size, fwhm=fwhm)
    gauss_kernel = tf.tile(gauss_kernel, [1, 1, 1, num_angles])
    return Convolution(input, gauss_kernel, padding="SAME")


def makeGaussian(size, fwhm=3, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    gauss = np.exp(-4 * np.log(2) * ((x - x0)**2 + (y - y0)**2) / fwhm**2)
    #plt.imshow(gauss)
    gauss = tf.constant(gauss, dtype=tf.float32)[:-1, :-1, tf.newaxis,
                                                 tf.newaxis]
    return gauss


def rotate(input, theta):
    rot_layer = tf.keras.layers.RandomRotation((theta, theta),
                                               fill_mode='constant',
                                               interpolation='bilinear',
                                               seed=None,
                                               fill_value=0.0)
    return rot_layer(input)


def derivate(input, num_angles):
    derivative_kernel = tf.constant(np.array([[1, -2, 1]]).transpose(),
                                    dtype=tf.float32)
    derivative_kernel = derivative_kernel[:, :, tf.newaxis, tf.newaxis]
    derivative_kernel = tf.tile(derivative_kernel, [1, 1, 1, num_angles])
    derivative_mat = Convolution(input, derivative_kernel, padding='VALID')
    return derivative_mat


def square(input):
    return tf.keras.layers.Activation(tf.math.square)(input)


def summ(input):
    return tf.transpose(tf.math.reduce_sum(input, axis=1, keepdims=True),
                        [0, 2, 1, 3])


def add_noize(input, std, n):
    input2 = tf.tile(input, [n, 1, 1, 1])
    noise = 1 + tf.random.normal(
        shape=tf.shape(input2), mean=0.0, stddev=std, dtype=tf.float32)
    return input2 * noise


def round_fun(shape, centr, fun):
    center = np.array(centr)
    mas = np.zeros(shape)
    for x in range(shape[0]):
        for y in range(shape[1]):
            p = np.array((x, y))
            mas[x, y] = fun(np.linalg.norm(p - center))
    return mas


def hat(x, r):
    if abs(x) < r:
        resalt = math.exp(1 / 8 / ((x / r)**2 - 1**2))
    else:
        resalt = 0
    return resalt


def loss_fun(input, alf):
    return tf.keras.layers.Activation(lambda x: tf.math.sin(
        alf * tf.math.minimum(tf.math.square(x), tf.math.square(np.pi / 2))))(
            input)


def sum_losses(input):
    trans = 1 - input
    return 1 - tf.reduce_prod(trans, axis=1, keepdims=False)


def visual_for_test(ten, fun='img'):
    mas = ten.numpy()
    n_ang = mas.shape[-1]
    l = mas.shape[0]
    l2 = len(mas.shape)
    angls = np.linspace(0, 180, n_ang, endpoint=False)
    dic = {}
    ldic = {}
    for i in range(n_ang):
        dic2 = {}
        ldic2 = {}
        for j in range(l):
            if l2 == 4:
                dic2[j] = mas[j, :, :, i]
            else:
                dic2[j] = mas[j, :, i]
            ldic2[j] = np.amax(mas)
        dic[(0, angls[i])] = dic2
        ldic[(0, angls[i])] = ldic2
    if fun == 'img':
        fun = lambda x, mas: x.imshow(mas)
    if fun == 'plt':
        fun = lambda x, mas: x.plot(mas)
    ds.show_gerd(dic, fun, ldic)


def fiber_real_sim(pressure_mat, config, seed):
    np.random.seed(seed)
    geo = config['env']['sen_geometry']
    phys = config['env']['phys']
    n_angles = geo['n_angles']
    m = config['sim']['random_rot']
    if m == 1:
        m = None
    x = geo['distanse']
    fwhm = phys['fiber_sensibility']['value']
    kernl_size = phys['fiber_sensibility']['accuracy']
    fwhm = fwhm / x
    n = pressure_mat.shape[1]
    kernl_size = min(int(kernl_size * fwhm), n)
    alf = phys['kof']
    test = config['sim']['test_mod']

    n_images = pressure_mat.shape[0]
    X = pressure_mat.shape[1]
    Y = pressure_mat.shape[2]

    # pressure_mat = tf.constant(pressure_mat,dtype=tf.float32)
    pressure_mat2 = pressure_mat[:, :, :, tf.newaxis]
    if m == None:
        pressure_mat_angl = pressure_mat2
        m = 1
    else:
        pressure_mat3 = tf.tile(pressure_mat2, [m, 1, 1, 1])
        pressure_mat_angl = tf.keras.layers.RandomRotation(
            (0, 1),
            fill_mode='constant',
            interpolation='bilinear',
            seed=None,
            fill_value=0.0)(pressure_mat3)
    pressure_tensor = pressure_mat_angl[:, :, :, 0]
    rotated_array = []
    for i in range(n_angles):
        rot_mat = rotate(pressure_mat_angl, i / n_angles / 2)
        rotated_array.append(rot_mat)
    rot_tensor = tf.concat(rotated_array, axis=-1)
    if test:
        print('after_fiber_rot')
        visual_for_test(rot_tensor)
    # pressure_tensor2 = tf.slice(
    #     pressure_tensor, [0, int(X / 6.0), int(Y / 6.0)],
    #     [n_images * m,
    #      int(X * (1.0 - 2.0/6.0)),
    #      int(Y * (1.0 - 2.0/6.0))])
    # sliced_tensor = tf.slice(rot_tensor,
    #                          [0, int(X / 6.0), int(Y / 6.0), 0], [
    #                              n_images * m,
    #                              int(X * (1.0 - 2.0/6.0)),
    #                              int(Y * (1.0 - 2.0/6.0)), n_angles
    #                          ])
    sliced_tensor = rot_tensor
    if test:
        print('after_slise')
        visual_for_test(sliced_tensor)
    blured_mat = gauss_blur(sliced_tensor,
                            n_angles,
                            kern_size=kernl_size,
                            fwhm=fwhm)
    if test:
        print('after_blur')
        visual_for_test(blured_mat)
    sq_deriv_tensor = loss_fun(derivate(blured_mat, n_angles), alf)
    if test:
        print('loss_fun')
        visual_for_test(sq_deriv_tensor)
    sum_tensor = sum_losses(sq_deriv_tensor)
    if test:
        print('sum_loss')
        visual_for_test(sum_tensor, fun='plt')
    std = phys['reletive_noize']
    delt = phys['noize']
    signal = tf.random.normal((n_images, n, n_angles), mean=1,
                              stddev=std) * sum_tensor + tf.random.normal(
                                  (n_images, n, n_angles), mean=0, stddev=delt)
    if test:
        print('signal')
        visual_for_test(signal, fun='plt')
    return signal, pressure_tensor


def sim_on_gpu(mas, test_size, batch_size, config, seed):
    n_del = config['env']['sen_geometry']['n_spl']
    dataset = tf.data.Dataset.from_tensor_slices(mas[0:-test_size])
    batches = dataset.batch(batch_size, drop_remainder=False)
    dataset_test = tf.data.Dataset.from_tensor_slices(mas[-test_size:])
    batches_test = dataset_test.batch(batch_size, drop_remainder=False)

    input = []
    output = []
    for batch in batches:
        input1, output1 = fiber_real_sim(batch, config, seed)
        input1 = input1[:, ::n_del, :]
        input1 = tf.tile(input1, [1, n_del, 1])
        input.append(input1)
        output.append(output1)
    input = np.concatenate(input)
    output = np.concatenate(output)

    input_test = []
    output_test = []
    for batch in batches_test:
        input_test1, output_test1 = fiber_real_sim(batch, config, seed)
        input_test1 = input_test1[:, ::n_del, :]
        input_test1 = tf.tile(input_test1, [1, n_del, 1])
        input_test.append(input_test1)
        output_test.append(output_test1)
    input_test = np.concatenate(input_test)
    output_test = np.concatenate(output_test)
    return input, output, input_test, output_test
