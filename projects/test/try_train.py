import h5py
from lightning import pytorch as pl

# DeepCleanDataset
from train.cli import AframeCLI as dc_cli
from train.data import DeepCleanDataset as dc_dataset

# DeepClean
from train.model import DeepClean as dc_model
from train.architectures import Autoencoder
from train.metrics import OnlinePsdRatio, PsdRatio

# Logging
from utils.logging import configure_logging
from train.callbacks import ModelCheckpoint, PsdPlotter

# print(dir(dc_cli))
# print(dir(dc_dataset))
# print(type(dc_dataset))

# strain and witness channels
data_file = "/home/kamalan/dc-demo/data/K-K1_lldata-1369291863-16384.hdf5"
chs = [
    "K1:CAL-CS_PROC_DARM_STRAIN_DBL_DQ",
    "K1:PEM-MIC_BS_FIELD_BS_Z_OUT_DQ",
    "K1:PEM-MIC_BS_BOOTH_BS_Z_OUT_DQ",
    "K1:PEM-MIC_BS_TABLE_POP_Z_OUT_DQ",
    "K1:PEM-MIC_IXC_BOOTH_IXC_Z_OUT_DQ",
    "K1:PEM-MIC_IXC_FIELD_IXC_Z_OUT_DQ",
    "K1:PEM-MIC_IYC_BOOTH_IYC_Z_OUT_DQ",
    "K1:PEM-MIC_OMC_BOOTH_OMC_Z_OUT_DQ",
    "K1:PEM-MIC_OMC_TABLE_AS_Z_OUT_DQ",
    "K1:PEM-MIC_TMSX_TABLE_TMS_Z_OUT_DQ",
    # "K1:PEM-VOLT_AS_TABLE_GND_OUT_DQ",
    # "K1:PEM-VOLT_IMCREFL_TABLE_GND_OUT_DQ",
    # "K1:PEM-VOLT_ISS_TABLE_GND_OUT_DQ",
    # "K1:PEM-VOLT_OMC_CHAMBER_GND_OUT_DQ",
    # "K1:PEM-VOLT_PSL_TABLE_GND_OUT_DQ",
    # "K1:PEM-VOLT_REFL_TABLE_GND_OUT_DQ",
]

with h5py.File(data_file, "r") as f:
    keys = list(f.keys())
# print(f[keys[0]].attrs['dx'])
# print(1/f[keys[0]].attrs['dx'])

# Set the DeepCleanDataset
t_dataset = dc_dataset(
    fname=data_file,
    channels=chs,
    kernel_length=8,
    freq_low=55,
    freq_high=65,
    batch_size=512,
    train_duration=4096,
    test_duration=8192,
    valid_frac=0.33,
    train_stride=0.0625,
    inference_sampling_rate=64,
    start_offset=0,
    filt_order=8,
)

# print(t_dataset.strain_channel)
# print(t_dataset.witness_channels)
# print(t_dataset.num_witnesses)
print(t_dataset.kernel_size)

# t_dataset.load_timeseries("train")

# Set the DeepClean
t_model = dc_model(
    arch=Autoencoder,
    loss=PsdRatio,
    metric=OnlinePsdRatio,
    patience=None,
    save_top_k_models=10,
)

# Set the DeepCelanCLI
tcli = dc_cli(
    model_class=t_model,
    datamodule_class=t_dataset,
    seed_everything_default=23984,
    run=False,
    parser_kawrgs={"default_env": True},
    save_config_kwargs={"overwite": True},
    args=args,
)
