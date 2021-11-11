import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import wandb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    """Define zero as midpoint of colour map.

    See https://stackoverflow.com/a/50003503/3790116.
    """

    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(
            0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(
            1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [
            normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))


def navier_stokes_heatmap(vals, name, range=None):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    if range is None:
        vmax = np.abs(vals).max()
        vmin = -vmax
    else:
        vmin = -range
        vmax = range
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    cmap = plt.get_cmap('RdBu')
    im = ax.imshow(vals, interpolation='bilinear', norm=norm, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    fig.tight_layout()
    return fig


def multi_heatmap(arrs, names, range=None, base_size=3.0, dpi=100):
    n_ims = len(arrs)
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

