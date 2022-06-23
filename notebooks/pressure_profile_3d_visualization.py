# %%
import matplotlib.pyplot as plt
import numpy as np

from avi_r import AVIReader
from PIL import Image

video = AVIReader('data/videos_from_isaac/0/force.avi')
plt.rcParams["figure.figsize"] = (8, 6)

# %%
n = 0
for x in video:
    n += 1
print(n)

# %%
# %matplotlib notebook
res = []

# # визуализация
# for i, pic in enumerate(video):
#     if i == 260:
#         pic = pic.numpy()
#         fig = plt.figure()
#         ax = fig.add_subplot(projection='3d')
#         X, Y = np.meshgrid(np.arange(pic.shape[0]), np.arange(pic.shape[1]))
#         print(pic.shape)
#         ax.plot_surface(X, Y, pic[:, :, 0], cmap=plt.cm.Spectral)
#         # plt.imshow(pic.numpy())
#         # plt.title(i)
#         plt.show()
#         # break
# %%
# # конверитаци в numpy
# for i, pic in enumerate(video):

#     image = Image.fromarray(pic.numpy()[:, :, 0]).resize((64, 64))
#     res.append(np.array(image, dtype=np.float32)[np.newaxis, :, :])
#     # if i == 2:
#         # break

# # plt.imshow(pic.numpy())
# video.close()

# res = np.concatenate(res)
# np.save('data/pressure/pic/0.npy', res)
# # # %%
# # plt.plot(np.arange(1, 10, 0.1))
# # plt.show()
# %%

N = 260

for i, pic in enumerate(video):
    if i == N:
        pic = np.array(pic.numpy(), dtype=np.int64)
        break

# plt.imshow(pic)
# plt.show()

row = pic[1750 * len(pic) // 2048,
          900 * len(pic) // 2048:1500 * len(pic) // 2048, 0]
plt.plot(row, 'o-')
plt.show()

step = 2
for step in [1, 3, 10, 50]:
    der = (row[step:] - row[:-step]) / step
    der2 = (der[step:] - der[:-step]) / step
    plt.title("Second derivative with different steps")
    plt.plot(der2, 'o-', ms=5, label='step = ' + str(step))
plt.legend()
plt.show()
# %%
downscale = 5
plt.plot(row[::downscale])
plt.title("downscaling of picture in " + str(downscale) + " times")
plt.show()
# %%
plt.imshow(pic[::, ::])