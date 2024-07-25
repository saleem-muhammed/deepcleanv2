import os
import yaml
import torch
from ml4gw.transforms import ChannelWiseScaler
from utils.filt import BandpassFilter


class InferenceModel:
    def __init__(self, train_dir, sample_rate, device='cpu'):
        self.device = device
        
        # Load configuration from YAML
        with open(os.path.join(train_dir, "config.yaml"), 'r') as file:
            train_config = yaml.safe_load(file)
        
        num_witnesses = len(train_config['data']['channels']) - 1
        freq_low   = train_config['data']['freq_low']
        freq_high  = train_config['data']['freq_high']
        filt_order = int(train_config['data']['filt_order'])
        self.sample_rate = sample_rate

        # Load model
        self.model = torch.jit.load(os.path.join(train_dir, "model.pt")).to(device)
        
        # Load scalers
        self.y_scaler = ChannelWiseScaler()
        self.y_scaler.load_state_dict(torch.load(os.path.join(train_dir, "y_scaler.pt")))
        
        self.X_scaler = ChannelWiseScaler(num_witnesses)
        self.X_scaler.load_state_dict(torch.load(os.path.join(train_dir, "X_scaler.pt")))
        
        # Initialize bandpass filter
        self.bandpass = BandpassFilter(freq_low, freq_high, sample_rate, filt_order)
        
        # Set channels 
        channels = train_config['data']['channels']
        # temporary fix for saleem's purpose
        # channels[0] = 'H1:GDS-CALIB_STRAIN_CLEAN'
        self.channels = channels
