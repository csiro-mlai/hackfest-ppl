import os

import h5py
import numpy as np
from einops import rearrange, repeat
from torch.utils.data import DataLoader, Dataset, get_worker_info
from pytorch_lightning import LightningDataModule

from ..utils import resolve_path

"""
This dataset is packed as as (batch, x, y, time) i.e. batch first, time last.

There is some messiness here. on windows or macos the HDF5 connection will not work in multiprocess mode because the processes are spawned by `spawn` instead of `fork`, so we need to mess around with `load`/`unload` to prevent serialization errors.
If we had more time we could do this with some kind of pickling system I guess?
"""


class NavierStokesH5InstDatastore(LightningDataModule):
    """
    This "instantaneous" data loader loads up a time instant and its immediate history, plus subsequent steps up to n_horizon ahead.
    """
    name = 'navier_stokes_h5_inst'

    def __init__(self,
            data_path: str,
            ssr: int=1,
            n_steps: int=11,
            n_workers: int=2,
            batch_size: int=20,
            latent_key: str = 'f',
            forcing_key: str = '',
            param_key: str = ''
        ):
        """
        forcing_key denotes a time-varying forcing
        latent_key denotes a time-invariant latent param
        """
        super().__init__()
        self.n_workers = n_workers
        self.batch_size = batch_size

        self.train_dataset = NavierStokesInstDataset(
            NavierStokesTSDataset(
                data_path,
                'train',
                ssr,
                latent_key=latent_key,
                forcing_key=forcing_key,
                param_key=param_key
            ),
            n_steps)
        self.valid_dataset = NavierStokesInstDataset(
            NavierStokesTSDataset(
                data_path,
                'valid',
                ssr,
                latent_key=latent_key,
                forcing_key=forcing_key,
                param_key=param_key
            ),
            n_steps)
        self.test_dataset = NavierStokesInstDataset(
            NavierStokesTSDataset(
                data_path,
                'test',
                ssr,
                latent_key=latent_key,
                forcing_key=forcing_key,
                param_key=param_key
            ),
            n_steps)
        if self.n_workers>0:
            self.train_dataset.unload()
            self.valid_dataset.unload()
            self.test_dataset.unload()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            worker_init_fn=load if self.n_workers>0 else None,
            drop_last=False,
            pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # helps diags
            num_workers=self.n_workers,
            worker_init_fn=load if self.n_workers>0 else None,
            drop_last=False,
            pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # helps diags
            num_workers=self.n_workers,
            worker_init_fn=load if self.n_workers>0 else None,
            drop_last=False,
            pin_memory=True)


class NavierStokesH5TSDatastore(LightningDataModule):
    """
    This "instantaneous" data loader loads up a whole time series at once.
    """
    name = 'navier_stokes_h5_ts'

    def __init__(self,
            data_path: str,
            ssr: int=1,
            n_workers: int=2,
            batch_size: int=20,
            latent_key: str = 'f',
            forcing_key: str = '',
            param_key: str = ''
        ):
        """
        forcing_key denotes a time-varying forcing
        latent_key denotes a time-invariant latent param
        """
        super().__init__()
        self.n_workers = n_workers
        self.batch_size = batch_size

        # Keep h5 references so that the backing HDF store loads on demand
        self.train_dataset = NavierStokesTSDataset(
            data_path,
            'train',
            ssr,
            latent_key=latent_key,
            forcing_key=forcing_key,
            param_key=param_key
        )
        self.valid_dataset = NavierStokesTSDataset(
            data_path,
            'train',
            ssr,
            latent_key=latent_key,
            forcing_key=forcing_key,
            param_key=param_key
        )
        self.test_dataset = NavierStokesTSDataset(
            data_path,
            'train',
            ssr,
            latent_key=latent_key,
            forcing_key=forcing_key,
            param_key=param_key
        )
        if self.n_workers>0:
            self.train_dataset.unload()
            self.valid_dataset.unload()
            self.test_dataset.unload()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            worker_init_fn=load if self.n_workers>0 else None,
            drop_last=False,
            pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # helps diags
            num_workers=self.n_workers,
            worker_init_fn=load if self.n_workers>0 else None,
            drop_last=False,
            pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # helps diags
            num_workers=self.n_workers,
            worker_init_fn=load if self.n_workers>0 else None,
            drop_last=False,
            pin_memory=True)


class NavierStokesTSDataset(Dataset):
    """
    each item is a whole time series
    """
    def __init__(self, data_path, data_key, ssr, latent_key, forcing_key, param_key):
        self.data_path = data_path
        self.data_key = data_key
        self.ssr = ssr
        self.latent_key = latent_key
        self.forcing_key = forcing_key
        self.param_key = param_key
        self.load()

    def load(self):
        """
        Set up the actual HDF5 file.
        This might need to be repeated if we spawn workers.
        """
        self.data = h5py.File(resolve_path(self.data_path))[self.data_key]
        self.B = self.data['u'].shape[0]

    def unload(self):
        """
        remove the HDF5 which cannot be serialized.
        """
        self.data = None

    def __len__(self):
        return self.B

    def tslen(self):
        """
        Number of steps in ts, based on batch-first, time-last order of the N-S simulator.
        """
        return self.data['u'].shape[-1]

    def __getitem__(self, b):
        elem = {
            'x': self.data['u'][b, ::self.ssr, ::self.ssr, :],
        }
        if len(self.latent_key):
            elem['latent'] = self.data[self.latent_key][b, ::self.ssr, ::self.ssr]
        # slightly confusing because sometime something called 'f' in the HDF file might be treated as a latent for an experiment
        if len(self.forcing_key):
            elem['f'] = self.data[self.forcing_key][b, ::self.ssr, ::self.ssr, :]
        if len(self.param_key):
            elem['param'] = self.data[self.param_key][b]
        return elem


class NavierStokesInstDataset(Dataset):
    """
    each item is a stretch of time series.
    This is very similar to the TS dataset, but with a different slicing strategy.
    """
    def __init__(self, ts_dataset, n_steps):
        self.n_steps = n_steps
        self.ts_dataset = ts_dataset
        self.load()

    def load(self):
        """
        Set up the actual HDF5 file
        """
        self.ts_dataset.load()
        self.B = len(self.ts_dataset)
        self.T = self.ts_dataset.tslen() - self.n_steps + 1

    def unload(self):
        """
        remove the HDF5 which cannot be serialized.
        """
        self.ts_dataset.unload()

    def __len__(self):
        return self.B * self.T

    def __getitem__(self, idx):
        """
        logic to select both example and timestep using a single scalar
        """
        b = idx // self.B
        t = idx % self.T
        ts = self.ts_dataset[b]
        if 'f' in ts:
            ts['f'] = ts['f'][..., t:t+self.n_steps][...]
        x = ts['x']
        ts['x'] = x[..., t:t+self.n_steps][...]
        return ts


def load(worker_id):
    """
    see https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
    """
    worker_info = get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    dataset.load()
    # print(f'worker {worker_id} loaded, {worker_info}')
