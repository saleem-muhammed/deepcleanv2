import logging

import torch
from pathlib import Path
import numpy as np

from lightning import pytorch as pl
from ml4gw.dataloading import InMemoryDataset

from clean.frames import FrameCrawler, frame_it, parse_frame_name
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
        batch_size: int = 32,
        # clean_stride: float = 1,
        inference_sampling_rate: float = 2,
        kernel_size: float = 1,
        device : str = "cuda"
    ):
        super().__init__()
        self.__logger = logging.getLogger("DeepClean Dataset")
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.hoft_crawler    = FrameCrawler (Path(hoft_dir))
        self.witness_crawler = FrameCrawler (Path(witness_dir))

        self.frame_iterator = frame_it(self.hoft_crawler, 
                                       self.witness_crawler, 
                                       self.channels_ordered_list, 
                                       self.model.sample_rate)

        self.update()

    @property
    def strain_channel(self):
        return self.model.channels[0]

    @property
    def witness_channels(self):
        return sorted(self.model.channels[1:])

    @property
    def channels_ordered_list(self):
        return [self.strain_channel]+ self.witness_channels

    @property
    def num_witnesses(self):
        return len(self.model.channels) - 1

    @property
    def kernel_size(self):
        return int(self.hparams.kernel_size * self.model.sample_rate)
    
    @property
    def stride(self):
        return int(self.model.sample_rate/self.hparams.inference_sampling_rate)


    def update(self):
        
        # updates the data by adding a newer second and 
        # removing the very first in the buffer
        
        self.strain_fname, strain, witnesses = next(self.frame_iterator)
        self.prefix, self.t0, self.duration = parse_frame_name(self.strain_fname)
        self.y_inference = torch.Tensor(np.array(strain, dtype=np.float64))
        self.X_inference = torch.Tensor(witnesses)
                
        ## then normalize them
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
        )

        self.y_inference = InMemoryDataset(
            self.y_inference,
            kernel_size=self.kernel_size,
            batch_size=self.hparams.batch_size,
            stride = self.stride, 
            coincident=True,
            shuffle=False,
            device = self.hparams.device,
        )
