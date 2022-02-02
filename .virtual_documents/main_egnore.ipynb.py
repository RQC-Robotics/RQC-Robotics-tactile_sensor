# get_ipython().getoutput("git clone https://github.com/RQC-Robotics/RQC-Robotics-tactile_sensor")
# import sys
# sys.path.append('/content/RQC-Robotics-tactile_sensor')
import sensor_lib as sl


n_gauses=10
n_pic=100
n_fibers=4
n_rot=1
lerning_rate=1e-4
n_epochs=30


sl.generate_pressure_map(n_pic, n_fibers, x= 97, y = 97, part='generated_gauses/fresh_gauss'+str(n_gauses)+'_'+str(n_pic)+'.npy')


input, output, input_test, output_test = sl.sim_on_gpu('generated_gauses/fresh_gauss'+str(n_gauses)+'_'+str(n_pic)+'.npy', n_random_rot=n_rot, n_angles=n_fibers, batch_size_preproc=128*8,size=100,test_size=10)
input_shape = input.shape
output_shape = output.shape
dataset_b, dataset = sl.prepare_dataset_for_train(input, output, batch_size_fit_model=1024*2)
dataset_test_b, dataset_test = sl.prepare_dataset_for_train(input_test, output_test, batch_size_fit_model=1024*2)


model = sl.SensorNN4S(input_shape[1:3], output_shape[1:3])
model.build(input_shape)
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(lerning_rate), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])


with tf.device('/gpu:1'): 
    model.fit(dataset_b, epochs = n_epochs, verbose=1)


model.save('models_weights/model4S_'+str(n_fibers)+'fi.nn')


model.evaluate(dataset_test_b)


predictions = model.predict(dataset_test_b)
predictions.shape


with open('pred_4S_'+str(n_gauses)+'g_'+str(n_fibers)+'fi', 'wb') as f:
    np.save(f, predictions)
    np.save(f, output_test)


N = 24 #28 26 24 23 21 20
plt.imshow(predictions[N])
plt.show()
plt.imshow(output_test[N])
plt.show()
