import h5py
import numpy as np
import pickle 
from  gwpy.timeseries import TimeSeries
import os

from train.model import DeepClean
from train.architectures import Autoencoder
from train.metrics import OnlinePsdRatio, PsdRatio
from lightning import pytorch as pl
import logging


from ml4gw.transforms import ChannelWiseScaler

from utils.filt import BandpassFilter
from clean.frames import  FrameCrawler, Path, Buffer, frame_it
from clean.data import DeepCleanInferenceDataset
from clean.infer import OnlineInference

## all the paramters go here (will be taken from the config file)

# model parameters
path_to_model = "../test/test_run/lightning_logs/version_0/checkpoints/last.ckpt"
num_witnesses=21 
hidden_channels=[8,16,32,64]
sample_rate = 4096
fftlength = 2
freq_low = [55]
freq_high = [65]
filt_order = 8
inference_sampling_rate= 1
edge_pad= 0.25
filter_pad= 0.5
device = "cuda"

hoft_dir = "/home/muhammed.saleem/deepClean/deepclean/O3_60Hz/H1/pseudo_replay/llhoft/"
witness_dir = "/home/muhammed.saleem/deepClean/deepclean/O3_60Hz/H1/pseudo_replay/lldetchar/"

outdir = './outdir/cleaned_frames'

channels = ["H1:GDS-CALIB_STRAIN_CLEAN",
            "H1:PEM-CS_MAINSMON_EBAY_1_DQ",
            "H1:ASC-INP1_P_INMON",
            "H1:ASC-INP1_Y_INMON",
            "H1:ASC-MICH_P_INMON",
           "H1:ASC-MICH_Y_INMON",
           "H1:ASC-PRC1_P_INMON",
           "H1:ASC-PRC1_Y_INMON",
           "H1:ASC-PRC2_P_INMON",
           "H1:ASC-PRC2_Y_INMON",
           "H1:ASC-SRC1_P_INMON",
           "H1:ASC-SRC1_Y_INMON",
           "H1:ASC-SRC2_P_INMON",
           "H1:ASC-SRC2_Y_INMON",
           "H1:ASC-DHARD_P_INMON",
           "H1:ASC-DHARD_Y_INMON",
           "H1:ASC-CHARD_P_INMON",
           "H1:ASC-CHARD_Y_INMON",
           "H1:ASC-DSOFT_P_INMON",
           "H1:ASC-DSOFT_Y_INMON",
           "H1:ASC-CSOFT_P_INMON",
           "H1:ASC-CSOFT_Y_INMON"]
model = DeepClean.load_from_checkpoint(
    path_to_model, 
    arch=Autoencoder(num_witnesses=num_witnesses, 
            hidden_channels=hidden_channels),
    loss=PsdRatio(sample_rate = sample_rate,
            fftlength = fftlength,
            freq_low = freq_low,
            freq_high = freq_high),
    metric=OnlinePsdRatio(inference_sampling_rate= inference_sampling_rate,
            edge_pad= edge_pad,
            filter_pad= filter_pad,
            sample_rate= sample_rate,
            bandpass= BandpassFilter(freq_low, freq_high, sample_rate, filt_order),
            y_scaler= ChannelWiseScaler()),
    patience=None,
    save_top_k_models=10,
)


inference_dataset = DeepCleanInferenceDataset(
        hoft_dir = hoft_dir,
        witness_dir = witness_dir,
        channels = channels,
        freq_low = freq_low,
        freq_high = freq_high,
        sample_rate = sample_rate,
        filt_order = filt_order,
        device = device)


_, y, x = next(inference_dataset.frame_iterator)
_, y, x = next(inference_dataset.frame_iterator)

seg1_middle_y = y[4096:-4096]
seg1_middle_x = x[:,4096:-4096]

_, y, x = next(inference_dataset.frame_iterator)
seg2_first_y = y[:4096]
seg2_first_x = x[:,:4096]

diff_y = seg2_first_y - seg1_middle_y
diff_x = seg2_first_x - seg1_middle_x

print(min(diff_y), max(diff_y))
print(min(diff_x.flatten()), max(diff_x.flatten()))
