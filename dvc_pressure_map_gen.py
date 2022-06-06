import sensor_lib as sl
import tensorflow as tf
import numpy as np
import sys
from os.path import join as jn
import yaml

with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)
geo = config['env']['sen_geometry']
x = geo['x_len']
y = geo['y_len']
n_pic = config['dataset']['n_samples']
n_gaus = config['env']['pressure_profile']['n_gauses']
size_kof = config['env']['pressure_profile']['size_kof']

np.random.seed(config['random_seed'])
seeds = np.random.randint(0, 2**31, size=3)

vec_mat = tf.constant(sl.get_vec_mat(x, y), dtype=tf.float32)
vec_mat = tf.reshape(vec_mat, [-1, 2])
gaus_data = sl.gen_rand_cof(n_gaus * n_pic, x, y, size_kof,
                            seed=seeds[0])    # use random inside
gaus_data = tf.reshape(gaus_data, [n_pic, n_gaus, 5])
with open(config['env']['pressure_profile']['g_param_path'], 'wb+') as f:
    np.save(f, gaus_data)

bs_gpu = config['gengaus']['batch_size']
bs_save = config['gengaus']['save_bath_size']
dataset = tf.data.Dataset.from_tensor_slices(gaus_data)
batches = dataset.batch(bs_gpu, drop_remainder=False)

i = 0
n = int(bs_save / bs_gpu)
pictures = []
for batch in batches:
    picture = sl.generate_pictures(batch, vec_mat)
    pictures.append(picture)
    print('generated batch:', i)
    i += 1
    if i % n == 0:
        pictures = tf.concat(pictures, axis=0)
        pictures = tf.reshape(pictures, [-1, x, y])
        with open(jn(config['dataset']['pic_path'],
                     str(i//n - 1) + '.npy'), 'wb+') as f:
            np.save(f, pictures)
        pictures = []

if len(pictures) > 0:
    pictures = tf.concat(pictures, axis=0)
    pictures = tf.reshape(pictures, [-1, x, y])
    with open(jn(config['dataset']['pic_path'],
                 str(i // n) + '.npy'), 'wb+') as f:
        np.save(f, pictures)
