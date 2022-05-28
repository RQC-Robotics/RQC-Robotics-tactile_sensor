import numpy as np
# import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from os.path import join as jn


def show(mas, points, size=5):
    X = len(mas)
    Y = len(points)
    fig, axes = plt.subplots(Y, X)
    for N in points:
        i = 0
        for m in mas:
            axes[N, i].imshow(m.numpy()[N])
            i += 1
    fig.set_figwidth(size * X)
    fig.set_figheight(size * Y)
    plt.show()


def show_gerd(dic, fun, text_dic={}, size=(5, 5), save=False, name='pic'):

    def key_fun(x):
        if type(x) == type(''):
            return 100000
        else:
            return x

    a_key = set()
    x_key = set()
    y_key = set()
    for key in dic.keys():
        x, y = key
        x_key.add(x)
        y_key.add(y)
    val = next(iter(dic.values()))    # ???
    for k, v in val.items():
        a_key.add(k)
    a_key = sorted(list(a_key), reverse=False)
    x_key = sorted(list(x_key), key=key_fun, reverse=False)
    y_key = sorted(list(y_key), key=key_fun, reverse=True)
    X = len(x_key)
    Y = len(y_key)
    A = len(a_key)

    fig, axes = plt.subplots(X * A, Y, squeeze=False)
    for a in range(A):
        for y in range(Y):
            for x in range(X):
                coord = (a*X + x, y)
                key = (x_key[x], y_key[y])
                if key in dic:
                    pic = dic[key][a_key[a]]
                    fun(axes[coord], pic)
                if key in text_dic:
                    title = str(round(text_dic[key][a_key[a]], 3))
                else:
                    title = ''
                axes[coord].set_title(str(key) + ' ' + title)
    fig.set_figwidth(size[0] * Y)
    fig.set_figheight(size[1] * X * A)
    if save:
        fig.savefig(name + '.png')
    plt.show()


def show_datasets(dataSets):
    vals = []
    keys = []
    for key, val in dataSets.items():
        df = val.reset_index()
        del (df['index'])
        vals.append(df)
        keys.append(key)
    df = pd.concat(vals, keys=keys, axis=1)
    show_gerd(
        df.xs('pic', level=1, axis=1).to_dict(), lambda x, mas: x.imshow(mas),
        df.xs('loss', level=1, axis=1).to_dict())


def loss(mt, mto):
    square = tf.keras.layers.Activation(tf.math.square)
    sqrt = tf.keras.layers.Activation(tf.math.sqrt)

    total_force = tf.reduce_mean(mt, [1, 2])
    dif = mt - mto
    dif2 = square(dif)
    err2 = tf.reduce_mean(dif2, [1, 2])
    err = sqrt(err2)
    error = err / total_force
    return error.numpy()


def loss2(mt, mto):
    square = tf.keras.layers.Activation(tf.math.square)
    sqrt = tf.keras.layers.Activation(tf.math.sqrt)

    max = tf.math.maximum(mt, mto)
    max = tf.math.maximum(max, 0.001)
    dif = (mt-mto) / max
    dif2 = square(dif)
    err2 = tf.reduce_mean(dif2, [1, 2])
    err = sqrt(err2)
    return err.numpy()


def get_dic(n_ga_mas, n_fi_mas, path_true, path_pred, l=100000, interval=None):
    dic = {}
    if interval == None:
        l2 = np.load(path_pred + str(n_ga_mas[0]) + 'g_' + str(n_fi_mas[0]) +
                     'fi').size[0]
        interval = (l - l2, l)

    for n_ga in n_ga_mas:
        mas = np.load(path_true + str(n_ga) + '_' + str(l) + '.npy',
                      mmap_mode='r')
        m = mas[interval[0]:interval[1]]
        m = m.astype(np.float32)
        n_images, X, Y = m.shape
        mt = tf.constant(m, dtype=tf.float32)
        mt = tf.slice(
            mt, [0, int(X / 6.0), int(Y / 6.0)],
            [n_images,
             int(X * (1.0 - 2.0/6.0)),
             int(Y * (1.0 - 2.0/6.0))])
        m = mt.numpy()
        dic[(n_ga, 'pic', 'input')] = [m[i] for i in range(m.shape[0])]

        for n_fi in n_fi_mas:
            mo = np.load(path_pred + str(n_ga) + 'g_' + str(n_fi) + 'fi')
            mto = tf.constant(mo, dtype=tf.float32)
            dic[(n_ga, 'loss', n_fi)] = loss(mt, mto)
            dic[(n_ga, 'pic', n_fi)] = [mo[i] for i in range(mo.shape[0])]
    return dic


# def get_dic2(path_true, path_pred):
#     mas = np.load(path_true)
#     mo = np.load(path_pred)
