# %%
import matplotlib.pyplot as plt
import numpy as np

from avi_r import AVIReader
video = AVIReader('data/videos_from_isaac/0/force.avi')

# %%
n = 0
for x in video:
    n += 1
print(n)

# %%
# %matplotlib notebook
for i, pic in enumerate(video):
    if i == 260:
        pic = pic.numpy()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        X, Y = np.meshgrid(np.arange(pic.shape[0]), np.arange(pic.shape[1]))
        print(pic.shape)
        ax.plot_surface(X, Y, pic[:, :, 0], cmap=plt.cm.Spectral)
        # plt.imshow(pic.numpy())
        # plt.title(i)
        plt.show()
        # break

# plt.imshow(pic.numpy())
video.close()

# # %%
# plt.plot(np.arange(1, 10, 0.1))
# plt.show()
