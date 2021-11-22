
import matplotlib.pyplot as plt
from math import sqrt, ceil, floor
import numpy as np
from matplotlib.colors import Normalize


def multi_img_plot_time(x, batch=0, interval=1, n_cols=None, fsize=6, interpolation=None):
    """
    Plot multiple timesteps of an array (time last)
    """
    steps = range(0, x.shape[-1], interval)
    if n_cols is None:
        n_cols = len(steps)
    print(steps, len(steps), n_cols)
    n_rows = ceil(len(steps) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fsize * n_cols/n_rows, fsize));
    axes = axes.flatten()
    for ax in axes:
        ax.set_axis_off()
    for i, ax in zip(steps, axes):
        ax.imshow(x[batch,..., i], interpolation=interpolation)
    plt.tight_layout()
    return fig

def multi_img_plot_batch(x, interval=1, n_cols=None, fsize=6, interpolation=None):
    """
    Plot multiple batches of an array 
    """
    if len(x.shape) == 4:
        x = np.squeeze(x, -1)

    steps = range(0, x.shape[0], interval)
    if n_cols is None:
        n_cols = len(steps)
    n_rows = ceil(len(steps) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fsize * n_cols/n_rows, fsize));
    axes = axes.flatten()
    for ax in axes:
        ax.set_axis_off()
    for i, ax in zip(steps, axes):
        ax.imshow(x[i,...], interpolation=interpolation)
    plt.tight_layout()
    return fig

def img_plot(x, interval=1, n_cols=None, fsize=6, interpolation=None):
    """
    plot whatever image I can find
    """
    x = np.squeeze(x, )
    plt.imshow(x, interpolation=interpolation)
    plt.gca().set_axis_off()
    plt.tight_layout()
    return plt.gcf()


def multi_heatmap(arrs, names=None, range=None, base_size=3.0, dpi=100):
    n_ims = len(arrs)
    if names is None:
        names = range(n_ims)
    fig, axs = plt.subplots(1, n_ims, figsize=(base_size*n_ims, base_size), dpi=dpi)
    if range is None:
        vmax = max([np.abs(arr).max() for arr in arrs])
        vmin = -vmax
    else:
        vmin = -range
        vmax = range
    ims = []
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('RdBu')
    if n_ims == 1:
        # ffs matplotlib "helps" by flattening the array. single subplot is not a thing.
        iter_axs = [axs]
    else:
        iter_axs = axs

    for imi, (arr, name, ax) in enumerate(zip(arrs, names, iter_axs)):
        ax.set_axis_off()
        ax.set_title(name)
        im = ax.imshow(arr, interpolation='bilinear', norm=norm, cmap=cmap)
        ims.append(im)

    fig.colorbar(ims[0], ax=axs, orientation='vertical', shrink = 0.6)
    # fig.tight_layout()
    return fig

