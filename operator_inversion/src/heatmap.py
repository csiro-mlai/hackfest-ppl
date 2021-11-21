import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


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

