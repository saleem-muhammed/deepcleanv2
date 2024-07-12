import logging

import h5py
import torch
from pathlib import Path

from lightning import pytorch as pl
from ml4gw.dataloading import InMemoryDataset
from ml4gw.transforms import ChannelWiseScaler

from utils.filt import BandpassFilter
from clean.frames import FrameCrawler, frame_it, Buffer
from clean.model import InferenceModel



class DeepCleanInferenceDataset(pl.LightningDataModule):
    """
    Many parameters such as sample_rate are coming from elsewhere 
    but have been explicitly defined here. 
    This needs to be changed
    """
    def __init__(
        self,
        hoft_dir: str,
        witness_dir: str,
        model: InferenceModel, 
        batch_size: int = 1,
        clean_stride: float = 1,
        kernel_length: float = 3,
        device : str = "cuda"
    ):
        super().__init__()
        self.__logger = logging.getLogger("DeepClean Dataset")
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.stride = int(clean_stride * self.model.sample_rate)
                        
        self.hoft_crawler    = FrameCrawler (Path(hoft_dir))
        self.witness_crawler = FrameCrawler (Path(witness_dir))

        self.frame_iterator = frame_it(self.hoft_crawler, 
                                       self.witness_crawler, 
                                       self.model.channels, 
                                       self.model.sample_rate)

        self.update()
        

    @property
    def strain_channel(self):
        return self.model.channels[0]

    @property
    def witness_channels(self):
        return sorted(self.model.channels[1:])

    @property
    def num_witnesses(self):
        return len(self.model.channels) - 1

    @property
    def kernel_size(self):
        return int(self.hparams.kernel_length * self.model.sample_rate)


    def update(self):
        
        # updates the data by adding a newer second and 
        # removing the very first in the buffer
        
        self.strain_fname, strain, witnesses = next(self.frame_iterator)

        self.y_inference = torch.Tensor(strain)
        self.X_inference = torch.Tensor(witnesses)
                
        ## then normalize them
        self.y_inference = self.model.y_scaler(self.y_inference)
        self.X_inference = self.model.X_scaler(self.X_inference)
        
        ## batching, bandpassing and moving the data to
        self.X_inference = InMemoryDataset(
            self.X_inference,
            kernel_size=self.kernel_size,
            batch_size=self.hparams.batch_size,
            stride = self.stride,
            coincident=True,
            shuffle=False,
            device = self.hparams.device,
            #device=f"cuda:{self.trainer.device_ids[0]}",
        )

        self.y_inference = InMemoryDataset(
            self.y_inference,
            kernel_size=self.kernel_size,
            batch_size=self.hparams.batch_size,
            stride = self.stride, 
            coincident=True,
            shuffle=False,
            device = self.hparams.device,
            #device=f"cuda:{self.trainer.device_ids[0]}",
        )
