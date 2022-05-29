import numpy as np
# import os
import matplotlib.pyplot as plt
import torch
import pandas as pd
from matplotlib import cm


def visual_picture(tensor: torch.Tensor, n_pictures, dim=2):
    size = 5
    if dim == 2:
        pictures = tensor[:n_pictures, 0].cpu().numpy()
    else:
        pictures = tensor[0].cpu().numpy()
    X = n_pictures
    fig, axes = plt.subplots(1, X)
    for i in range(X):
        if dim == 2:
            axes[i].imshow(pictures[i])
            fig.colorbar(cm.ScalarMappable(), ax=axes[i])
        else:
            axes[i].plot(pictures[i])
    fig.set_figwidth(size * X)
    fig.set_figheight(size)
    plt.show()
