import numpy as np
# import os
import matplotlib.pyplot as plt
import torch
import pandas as pd
import matplotlib as mpl


def visual_picture(tensor: torch.Tensor, n_pictures, dim=2, size=(5, 5)):
    if dim == 2:
        pictures = tensor[:n_pictures, 0].cpu().numpy()
    else:
        pictures = tensor[0].cpu().numpy()
    X = n_pictures
    fig, axes = plt.subplots(1, X, sharey='all', squeeze=False)
    axes = axes[0]
    for i in range(X):
        if dim == 2:
            axes[i].imshow(pictures[i])
            norm = mpl.colors.Normalize(pictures[i].min(), pictures[i].max())
            fig.colorbar(mpl.cm.ScalarMappable(norm), ax=axes[i])
        else:
            axes[i].plot(pictures[i])
    fig.set_figwidth(size[0] * X)
    fig.set_figheight(size[1])
    plt.show()


def visual_table_of_pictures(data,
                             sample_titles,
                             y_titles,
                             visual_fun,
                             size=(5, 5)):
    '''
    Params:
    data : List[List[np.array 1D or 2D]] - first coord -- sample, second -- different views
    sample_titles: List[str]
    y_titles: List[str] in order top down
    visual_fun: List[plt funcs] for each row

    usage example:
    
    data= [[pred_pic[i + 7*j] for i in range(3)] for j in range(4)]
    x_titles = ["best", "random", "random", "worst"]
    y_titles = ["true", "predict", "signal"]
    visual_func = [lambda ax, pic: ax.imshow(pic)]*3
    visual_table_of_pictures(data, x_titles, y_titles, visual_func)
    '''
    X = len(data)
    Y = len(data[0])
    fig, axes = plt.subplots(Y, X, squeeze=False, sharex=True, sharey='row')
    for y in range(Y):
        axes[y][0].set_ylabel(y_titles[y], fontsize=20)
        for x in range(X):
            pic = data[x][y]
            visual_fun[y](axes[y][x], pic)
    for x in range(X):
        axes[0][x].set_title(sample_titles[x], fontsize=20)
    fig.set_figwidth(size[1] * X)
    fig.set_figheight(size[0] * Y)
    # plt.show()
