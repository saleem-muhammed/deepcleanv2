import os
import torch # type: ignore
from gwpy.timeseries import TimeSeries, TimeSeriesDict # type: ignore
import logging
import numpy as np # type: ignore
from datetime import datetime

#from train.model import DeepClean
#from train.architectures import Autoencoder
#from train.metrics import OnlinePsdRatio
#from ml4gw.transforms import ChannelWiseScaler
#from utils.filt import BandpassFilter
#from clean.data import DeepCleanInferenceDataset


class OnlineInference:
    """
    The top-level inference class
        - uses frame directories, pre-loaded models, 
        and output directories as inputs
        
        - get pre-processed data to feed to the model
        - produces predictions 
        - postprocess the prediction
        - writes the cleaned frames   
    """

    def __init__(self, dataset, model, outdir, device):
        self.__logger = logging.getLogger("DeepClean Online Inference")
        self.model = model
        self.device = device
        self.dataset = dataset
        self.outdir = outdir
        self.logdir = os.path.join(self.outdir, "logs")
        self.edge_pad = (0.2*self.model.sample_rate)
        self.filter_pad = int(0.8*self.model.sample_rate)
        self.current_log_file = None
        self.current_date = None

        if not os.path.exists(outdir):
            os.makedirs(self.outdir)
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        
    def predict(self):
        witness = list(self.dataset.X_inference)[0]
        self.pred = self.model.model(witness.to(self.device))

    def postprocess(self):
        """
            Performs reverse scaling and bandpass of the noise prediction
        """
        self.noise = self.model.y_scaler(self.pred.cpu(), reverse=True)
        self.noise = self.noise[self.edge_pad:-self.edge_pad]
        self.noise = self.model.bandpass(self.noise.cpu().detach().numpy())
        self.noise = torch.tensor(self.noise, device=self.device).flatten()
        self.noise = self.noise[self.filter_pad:-self.filter_pad]
        
        self.raw   = list(self.dataset.y_inference)[0].to(self.device).flatten()
        self.raw   = self.raw[self.filter_pad:-self.filter_pad]
        self.raw   = self.raw.to(self.device) 

        self.cleaned = self.raw - self.noise

        self.set_fname()

        self.ts = self.make_gwpy_TimeSeries(
                data = self.cleaned, 
                channel = self.dataset.strain_channel+"_DC")
        
        self.ts_raw   = self.make_gwpy_TimeSeries(
                 data = self.raw, 
                 channel = self.dataset.strain_channel)
        
        # ts_noise   = self.make_gwpy_TimeSeries(
        #         data = self.noise, 
        #         channel = self.dataset.strain_channel+"_pred")

    def compute_asdr(self):
        asd_cleaned  = self.ts.asd(fftlength=1,overlap=0.0,method='median')
        asd_raw      = self.ts_raw.asd(fftlength=1,overlap=0.0,method='median')
        asd_cleaned_inband = asd_cleaned.crop(self.model.freq_low, self.model.freq_high)
        asd_raw_inband = asd_raw.crop(self.model.freq_low, self.model.freq_high)
        asd_ratio = (asd_cleaned_inband/asd_raw_inband).value
        self.mean_asd_ratio = np.mean(asd_ratio)
        self.max_asd_ratio = max(asd_ratio)

        
    def write_logs(self):
        self.compute_asdr()
        
        current_date = datetime.utcnow().strftime('%Y-%m-%d')
        
        if current_date != self.current_date:
            if self.current_log_file:
                self.current_log_file.close()
                
            log_filename = os.path.join(self.logdir, f"asdr_{current_date}.log")
            self.current_log_file = open(log_filename, 'a')
            self.current_date = current_date
        
        log_entry = f"{self.output_t0}, {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} - Mean ASDR: {self.mean_asd_ratio}, Max ASDR: {self.max_asd_ratio}\n"
        self.current_log_file.write(log_entry)
        self.current_log_file.flush()

    def set_fname(self):
        self.output_t0 = self.dataset.t0
        self.fname = f"{self.dataset.prefix}-{int(self.output_t0)}-{int(self.dataset.duration)}.gwf"
        self.outfile_path = os.path.join(self.outdir, self.fname)

    def make_gwpy_TimeSeries(self, data, channel):
        data = np.array(data.cpu().numpy(), dtype=np.float64)
        ts = TimeSeries(data=data, t0=self.output_t0, sample_rate=self.model.sample_rate,
                channel=channel, unit='seconds') 
        return ts 
        

    def write(self):
        
        #cleaned = cleaned.cpu().numpy()


        # Create a TimeSeriesDict and add the TimeSeries objects to it
        ts_dict = TimeSeriesDict()
        ts_dict[self.dataset.strain_channel+"_DC"] = self.ts
        # ts_dict[self.dataset.strain_channel] = self.ts_raw
        # ts_dict[self.dataset.strain_channel+"_pred"] = ts_noise

        # Write the TimeSeriesDict to a file
        ts_dict.write(self.outfile_path)
        
        self.__logger.info(f"Data written to file: {self.outfile_path}")
        print(f"Writing cleaned frame {self.outfile_path}")
        
        
    def predict_and_write(self):
        self.predict()
        self.postprocess()
        self.write()
        self.write_logs()
