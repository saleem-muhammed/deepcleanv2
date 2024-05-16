import h5py
import numpy as np

from train.model import DeepClean
from train.architectures import Autoencoder
from train.metrics import OnlinePsdRatio, PsdRatio
from lightning import pytorch as pl

from ml4gw.transforms import ChannelWiseScaler

from utils.filt import BandpassFilter


path_to_model = "../test/test_run/lightning_logs/version_0/checkpoints/last.ckpt"

#print (help(DeepClean.load_from_checkpoint))

model = DeepClean.load_from_checkpoint(
    path_to_model, 
    arch=Autoencoder(num_witnesses=21, 
            hidden_channels=[8,16,32,64]),
    loss=PsdRatio(sample_rate = 4096,
            fftlength = 2,
            freq_low = [55],
            freq_high = [65]),
    metric=OnlinePsdRatio(inference_sampling_rate= 64,
            edge_pad= 0.25,
            filter_pad= 0.5,
            sample_rate= 4096,
            bandpass= BandpassFilter([55], [65], 4096, 8),
            y_scaler= ChannelWiseScaler()),
    patience=None,
    save_top_k_models=10,
)

"""
First we load a certain number of framefiles iteratively using DeepCleanDataset and 
"""
witness_data = DeepCleanDataset ()

model ()

#model.freeze()

#model = dc_instance.load_from_checkpoint(path_to_model)

print ("\n Code running till the end\n")

## load some data from file and 