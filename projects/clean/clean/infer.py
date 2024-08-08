import os
import torch # type: ignore
from gwpy.timeseries import TimeSeries, TimeSeriesDict # type: ignore
import logging
import numpy as np # type: ignore

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
        self.edge2crop = int(1.0* self.model.sample_rate)

        if not os.path.exists(outdir):
            os.makedirs(self.outdir)
        
    def predict(self):
        witness = list(self.dataset.X_inference)[0]
        self.pred = self.model.model(witness.to(self.device))

    def postprocess(self):
        """
            Performs reverse scaling and bandpass of the noise prediction
        """
        self.noise = self.model.y_scaler(self.pred.cpu(), reverse=True)
        self.noise = self.model.bandpass(self.noise.cpu().detach().numpy())
        self.noise = torch.tensor(self.noise, device=self.device).flatten()
        self.noise = self.noise[self.edge2crop:-self.edge2crop]
        
        self.raw   = list(self.dataset.y_inference)[0].to(self.device).flatten()
        self.raw   = self.raw[self.edge2crop:-self.edge2crop]
        self.raw   = self.raw.to(self.device) 

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
        self.set_fname()
        cleaned = self.raw - self.noise
        #cleaned = cleaned.cpu().numpy()

        ts = self.make_gwpy_TimeSeries(
                data = cleaned, 
                channel = self.dataset.strain_channel+"_DC")
        
        # ts_raw   = self.make_gwpy_TimeSeries(
        #         data = self.raw, 
        #         channel = self.dataset.strain_channel)
        
        # ts_noise   = self.make_gwpy_TimeSeries(
        #         data = self.noise, 
        #         channel = self.dataset.strain_channel+"_pred")

        # Create a TimeSeriesDict and add the TimeSeries objects to it
        ts_dict = TimeSeriesDict()
        ts_dict[self.dataset.strain_channel+"_DC"] = ts
        # ts_dict[self.dataset.strain_channel] = ts_raw
        # ts_dict[self.dataset.strain_channel+"_pred"] = ts_noise

        # Write the TimeSeriesDict to a file
        ts_dict.write(self.outfile_path)
        
        self.__logger.info(f"Data written to file: {self.outfile_path}")
        print(f"Writing cleaned frame {self.outfile_path}")
        
        
    def predict_and_write(self):
        self.predict()
        self.postprocess()
        self.write()
