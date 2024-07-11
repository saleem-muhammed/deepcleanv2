import h5py
import numpy as np
import pickle 
from  gwpy.timeseries import TimeSeries
import os
import torch

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

device = "cuda"
num_witnesses=21 
sample_rate = 4096
freq_low = [55]
freq_high = [65]
filt_order = 8

# model parameters
model_repo = "../../myprojects/60Hz-O3-MDC/results/lightning_logs/version_8/"
model    = torch.jit.load(os.path.join(model_repo, "model.pt")).to(device)

y_scaler = ChannelWiseScaler()
y_scaler.load_state_dict(
    torch.load(os.path.join(model_repo, "y_scaler.pt")))

X_scaler = ChannelWiseScaler(num_witnesses)
X_scaler.load_state_dict(
    torch.load(os.path.join(model_repo, "X_scaler.pt")))

bandpass= BandpassFilter(freq_low, freq_high, sample_rate, filt_order)


#hidden_channels=[8,16,32,64]
#fftlength = 2
#inference_sampling_rate= 1
#edge_pad= 0.25
#filter_pad= 0.5
#device = "cuda"

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




inference_dataset = DeepCleanInferenceDataset(
        hoft_dir = hoft_dir,
        witness_dir = witness_dir,
        channels = channels,
        freq_low = freq_low,
        freq_high = freq_high,
        sample_rate = sample_rate,
        y_scaler = y_scaler,
        X_scaler = X_scaler,
        filt_order = filt_order,
        device = device)


online_inference = OnlineInference(
        dataset = inference_dataset,
        model  = model,
        y_scaler = y_scaler, 
        bandpass = bandpass, 
        outdir = outdir,
        device = device)


for k in range(100):
    online_inference.predict_and_write()
    online_inference.dataset.update()
    #print(f"iteration {k}")
    






######

# loss=PsdRatio(sample_rate = sample_rate,
#         fftlength = fftlength,
#         freq_low = freq_low,
#         freq_high = freq_high)

# metric=OnlinePsdRatio(inference_sampling_rate= inference_sampling_rate,
#         edge_pad= edge_pad,
#         filter_pad= filter_pad,
#         sample_rate= sample_rate,
#         bandpass= BandpassFilter(freq_low, freq_high, sample_rate, filt_order),
#         y_scaler= y_scaler)



# model = DeepClean.load_from_checkpoint(
#     last_model_path, 
#     arch=Autoencoder(num_witnesses=num_witnesses, 
#             hidden_channels=hidden_channels),
#     loss=PsdRatio(sample_rate = sample_rate,
#             fftlength = fftlength,
#             freq_low = freq_low,
#             freq_high = freq_high),
#     metric=OnlinePsdRatio(inference_sampling_rate= inference_sampling_rate,
#             edge_pad= edge_pad,
#             filter_pad= filter_pad,
#             sample_rate= sample_rate,
#             bandpass= BandpassFilter(freq_low, freq_high, sample_rate, filt_order),
#             y_scaler= ChannelWiseScaler()),
#     patience=None,
#     save_top_k_models=10,
# )

