import h5py

# DeepCleanCLI
import torch
from lightning.pytorch.cli import LightningCLI
#from train.cli import AframeCLI as dc_cli

# DeepCleanDataset
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
    freq_low=[55],
    freq_high=[65],
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
class DeepCleanCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--verbose", type=bool, default=False)

        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.OneCycleLR)

        parser.link_arguments(
            "data.num_witnesses",
            "model.arch.init_args.num_witnesses",
            apply_on="instantiate",
        )

        # link data arguments to loss function
        parser.link_arguments(
            "data.sample_rate",
            "model.loss.init_args.sample_rate",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.freq_low",
            "model.loss.init_args.freq_low",
            apply_on="parse",
        )
        parser.link_arguments(
            "data.freq_high",
            "model.loss.init_args.freq_high",
            apply_on="parse",
        )

        # link data arguments to metric
        parser.link_arguments(
            "data.inference_sampling_rate",
            "model.metric.init_args.inference_sampling_rate",
            apply_on="parse",
        )
        parser.link_arguments(
            "data.sample_rate",
            "model.metric.init_args.sample_rate",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.bandpass",
            "model.metric.init_args.bandpass",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.y_scaler",
            "model.metric.init_args.y_scaler",
            apply_on="instantiate",
        )

        # link optimizer and scheduler args
        parser.link_arguments(
            "data.steps_per_epoch",
            "lr_scheduler.steps_per_epoch",
            apply_on="instantiate",
        )
        parser.link_arguments("optimizer.lr", "lr_scheduler.max_lr")
        parser.link_arguments("trainer.max_epochs", "lr_scheduler.epochs")

def main(args=None):
    tcli = DeepCleanCLI(
        model_class=dc_model,
        datamodule_class=dc_dataset,
        seed_everything_default=23984,
        run=False,
        parser_kwargs={"default_env": True},
        save_config_kwargs={"overwite": True},
        args=args,
    )

    # log_dir = tcli.trainer.logger.log_dir or tcli.trainer.logger.save_dir
    # if not log_dir.startswith("s3://"):
    #     os.makedirs(log_dir, exist_ok=True)
    #     log_file = os.path.join(log_dir, "train.log")
    #     configure_logging(log_file)
    # else:
    #     configure_logging()
    # # tcli.trainer.fit(tcli.model, tcli.datamodule)

    # tcli.trainer.fit(tcli.model, tcli.datamodule)
    # if tcli.datamodule.hparams.test_duration > 0:
    #     tcli.trainer.test(tcli.model, tcli.datamodule)


if __name__ == "__main__":
    main()
