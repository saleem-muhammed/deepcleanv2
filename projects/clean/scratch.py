import numpy as np
import yaml
from gwpy.timeseries import TimeSeries, TimeSeriesDict
import torch
from glob import glob
import os
from clean.model import InferenceModel
from ml4gw.dataloading import InMemoryDataset



## all the paramters go here (will be taken from the config file)

clean_config = "/home/muhammed.saleem/deepClean/deepcleanv2/myprojects/60Hz-O3-MDC/config_clean.yaml"

with open(clean_config, 'r') as file:
        config = yaml.safe_load(file)

model = InferenceModel(config['train_dir'], config['sample_rate'], config['device'])

""" Load X_inference as required and resample to the target rate"""
hoft_files = glob(os.path.join(config['hoft_dir'], "*.gwf"))
woft_files = glob(os.path.join(config['witness_dir'], "*.gwf"))
start = 1250919100
end = start + 8

ts_dict = TimeSeriesDict.read(
        woft_files, 
        channels = model.channels[1:], 
        start=start, end=end)

# Resample each time series to the specified frequency
resampled_ts_dict = ts_dict.resample(model.sample_rate)
X_inference = np.column_stack([resampled_ts_dict[channel].value for channel in model.channels[1:]])

''

""" Load y_inference as required and resample to the target rate"""

X_inference = InMemoryDataset(
        X_inference,
        kernel_size=8,
        batch_size=1,
        stride = 1, 
        coincident=True,
        shuffle=False,
        device = config['device']
)

y_inference = InMemoryDataset(
        y_inference,
        kernel_size=8,
        batch_size=1,
        stride = 1, 
        coincident=True,
        shuffle=False,
        device = config['device']
)


witness = list(X_inference)[0]
pred = model.model(witness.to(config['device']))
edge2crop = int(1.0* model.sample_rate)
ts_dict = TimeSeriesDict()

noise = model.y_scaler(pred.cpu(), reverse=True)
noise = model.bandpass(noise.cpu().detach().numpy())
noise = torch.tensor(noise, device=config['device']).flatten()
noise = noise[edge2crop:-edge2crop]
        
raw   = list(y_inference)[0].to(config['device']).flatten()
raw   = raw[edge2crop:-edge2crop]
raw   = raw.to(config['device']) 

cleaned = raw - noise


output_t0 = start
ts = TimeSeries(data=np.array(cleaned.cpu().numpy(), dtype=np.float64), 
                t0=output_t0, 
                sample_rate=model.sample_rate,
                channel="CLEANED", unit='seconds') 
ts_dict["CLEANED"] = ts

ts_raw = TimeSeries(data=np.array(raw.cpu().numpy(), dtype=np.float64), 
                t0=output_t0, 
                sample_rate=model.sample_rate,
                channel="UNCLEAN", unit='seconds') 
ts_dict["UNCLEAN"] = ts_raw


ts_noise = TimeSeries(data=np.array(noise.cpu().numpy(), dtype=np.float64), 
                t0=output_t0, 
                sample_rate=model.sample_rate,
                channel="PRED", unit='seconds') 
ts_dict["PRED"] = ts_noise

ts_dict.write("debug.gwf")
        



# import pickle 
# import os
# import h5py

# from train.model import DeepClean
# from train.architectures import Autoencoder
# from train.metrics import OnlinePsdRatio, PsdRatio
# from lightning import pytorch as pl
# import logging


# from ml4gw.transforms import ChannelWiseScaler

# from utils.filt import BandpassFilter
# from clean.frames import  FrameCrawler, Path, Buffer, frame_it
# from clean.data import DeepCleanInferenceDataset
# from clean.infer import OnlineInference
