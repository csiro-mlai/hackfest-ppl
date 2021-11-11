"""
Reference code for point estimates by predictive error minimisation.

Currently keeping it around so that we can use some of the classes here but I think we can keep this most;ly in a notebook.
"""

from typing import Any, Dict, Tuple, Optional
from math import sqrt

import torch
from torch.nn.modules.loss import MSELoss
import torch.fft as fft
import torch.nn as nn
from torch.optim import AdamW

from .fourier_2d_generic import SimpleBlock2dGeneric
from .datamodules.navier_stokes_h5 import NavierStokesH5InstDatastore
from .utils import resolve_path
from torch.linalg import vector_norm, matrix_norm


class Fourier2dMapping(nn.Module):
    """
    Does not work because I tried to do something fancy with parameterizations.
    TODO.
    """

    def __init__(self, modes: int=20, dims: Tuple[int, int]=(256,256)):
        super().__init__()
        self.modes = modes  # maybe just normalize the weights?
        self.dims = dims

    def forward(self, X):
        """
        map from complex inputs on a half space to real inputs on a full space
        """
        print("X", X.shape, X.dtype)
        return fft.irfft2(X, s=self.dims, norm="ortho")

    def right_inverse(self, Xp):
        """
        map from real inputs on a full space to complex inputs on a half space
        """
        return fft.rfft2(Xp, s=self.dims, norm="ortho")


class NaiveLatent(nn.Module):
    def __init__(self,
            process_predictor: "nn.Module",
            dims: Tuple[int, int]=(256,256),
            n_batch: int=1):
        super().__init__()
        self.dims = dims
        self.process_predictor = process_predictor
        ## Do not fit the process predictor weights
        for param in self.process_predictor.parameters():
            param.requires_grad = False
        self.latent = nn.Parameter(
            torch.zeros(
                (n_batch, *dims),
                dtype=torch.float32
            )
        )

    def weights_init(self):
        self.latent.data.normal_(0.0, 0.01)

    def forward(self, batch):
        #copy
        batch = dict(**batch)
        batch['latent'] = self.latent
        return self.process_predictor(batch)



def fit(
        batch,
        model,
        loss_fn,
        optimizer,
        n_iter:int=20,
        check_int:int=1,
        clip_val: Optional[float] = None,
        callback = lambda *x: None,
        # pen_0: float = 0.0,
        pen_1: float = 0.0,
        pen_f: float = 0.0,
        stop_on_truth: bool = False,
        diminishing_returns=1.1,):
    model.train()
    model.weights_init()
    prev_loss_v = 10^5
    prev_error = 10^5
    prev_relerr = 10^3
    big_losses = []
    big_loss_fn = MSELoss(reduction='none')
    big_scale = big_loss_fn(torch.zeros_like(batch['latent']), batch['latent']).mean((1,2))
    scale = loss_fn(torch.zeros_like(batch['latent']), batch['latent']).item()
    for i in range(n_iter):
        # Compute prediction error
        pred = model(batch)
        loss = loss_fn(pred['forecast'], batch['y'])

        # if pen_0 > 0.0:
            # loss += model.latent.square().mean()* pen_0
        if pen_1 > 0.0:
            loss += model.latent.diff(dim =-1).square().mean() * pen_1
            loss += model.latent.diff(dim =-2).square().mean() * pen_1
        if pen_f > 0.0:
            loss += (model.latentf.abs().square()).mean() * pen_1

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        if clip_val is not None:
            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_value_(group["params"], clip_val)

        optimizer.step()
        # sch = self.lr_schedulers()
        # sch.step()

        if i % check_int == 0 or i==n_iter-1:
            with torch.no_grad():
                # recalc without penalties
                loss_v = loss_fn(pred['forecast'], batch['y']).item()
                if loss_v > diminishing_returns * prev_loss_v and i> 15:
                    print("Early stopping at optimum")
                    break
                prev_loss_v = loss_v
                error = loss_fn(model.latent, batch['latent']).item()
                if error > diminishing_returns * prev_error and stop_on_truth:
                    print("Early stopping at minimum prediction error")
                    break
                prev_error = error
                relerr = sqrt(error/scale)
                ##
                
                big_loss_v = big_loss_fn(pred['forecast'], batch['y']).mean((1,2,3))
                print(big_loss_v.shape)
                big_error = big_loss_fn(model.latent, batch['latent']).mean((1,2))
                big_relerr = torch.sqrt(big_error/scale)
                big_losses.append(dict(
                    big_loss=big_loss_v.detach().cpu().numpy(),
                    big_error = big_error.detach().cpu().numpy(),
                    relerr=big_relerr.detach().cpu().numpy()
                ))

                print(
                    f"loss: {loss:.3e}, error: {error:.3e}, relerror: {relerr:.3e} [{i:>5d}/{n_iter:>5d}]")
                callback(model, i, loss_v, error, loss_fn, batch, pred)

    loss_v = loss.item()
    error = loss_fn(model.latent, batch['latent']).item()
    scale = loss_fn(torch.zeros_like(batch['latent']), batch['latent']).item()
    relerr = sqrt(error/scale)
    print(
        f"loss: {loss:.3e}, error: {error:.3e}, relerror: {relerr:.3e} scale: {scale:.3e}[{i:>5d}/{n_iter:>5d}]")

    return loss_v, error, relerr, scale, big_losses, big_scale.detach().cpu().numpy()


def main(
        fwd_state_dict_path: str= '${SM_MODEL_DIR}/history_matching/adequate_checkpoint/fwd-epoch=19-step=26399-valid_loss=0.00000.ckpt',
        fwd_args: Dict[str, Any]={
            'modes1': 16,
            'width': 24,
            'n_layers': 4,
            'n_history': 2,
            'param': False,
            'forcing': False,
            'latent': True,
        },
        ds_args: Dict[str, Any]={
            'data_path': '${FNO_DATA_ROOT}/navier-stokes/grf_forcing_mini.h5',
            'ssr': 1,
            'n_history': 2,
            'n_horizon': 1,
            'batch_size': 20,
            'latent_key': 'f',
            'forcing_key': '',
            'param_key': '',
            'n_workers': 0,
        },
        dims: Tuple[int, int]=(256,256),
        n_batch: int = 1,
        example_i: int = 1,
        clip_val: Optional[float] = None,
        #  noise_std: float = 0.0
        devkey: str = 'cpu',
        lr: float = 0.0025,
        weight_decay: float = 0.0,
        f_weight: float = 0.0,
        callback = lambda *x: None,
        n_iter: int = 20,
        check_int: int = 1,
        mapping: str = 'identity',
        # pen_0: float = 0.0,
        pen_1: float = 0.0,
        pen_f: float = 0.0,
        **kwargs
        ):
    """
    Set up a pytorch optimisation which infers a latent field.
    """
    device = torch.device(devkey)
    datastore = NavierStokesH5InstDatastore(
        **ds_args
    )
    dataset = datastore.train_dataset
    batch = dataset[example_i]
    # the unsqueeze here makes that datapoint  into a singleton batch
    batch = dict([
        (key, torch.tensor(val).to(device).unsqueeze(0)) for key, val in batch.items()
    ])

    pp_state_dict = torch.load(
        resolve_path(fwd_state_dict_path),
        map_location=device
    )
    process_predictor = SimpleBlock2dGeneric(
        **fwd_args
    )
    process_predictor.load_state_dict(
        pp_state_dict
    )
    # model = NaiveLatent(
    #     process_predictor,
    #     dims=dims,
    #     n_batch=n_batch)
    # if mapping == 'fourier':
    #     parametrize.register_parametrization(
    #         model, "latent",
    #         Fourier2dMapping(dims=dims))
    # elif mapping == 'weighted_fourier':
    #     parametrize.register_parametrization(
    #         model, "latent",
    #         WeightedFourier2dMapping(
    #             dims=dims, penalty=f_weight)
    #     )

    if mapping == 'identity':
        model = NaiveLatent(
            process_predictor,
            dims=dims,
            n_batch=n_batch)
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay)
    elif mapping == 'fourier':
        raise NotImplementedError("plain Fourier not implemented")
    elif mapping == 'weighted_fourier':
        model = WeightedFourierLatent(
            process_predictor,
            dims=dims,
            n_batch=n_batch,
            f_weight=f_weight)
        optimizer = AdamWC(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"what even is {mapping}?")

    model.to(device)

    # loss_fn = LpLoss(size_average=True)
    loss_fn = nn.MSELoss()

    return model, fit(
        batch, model,
        loss_fn,
        optimizer,
        n_iter=n_iter,
        check_int=check_int,
        clip_val=clip_val,
        callback=callback,
        # pen_0=pen_0,
        pen_1=pen_1,
        pen_f=pen_f,
        **kwargs
    ), {'ds': datastore, 'batch': batch}

if __name__ == '__main__':
    main()

