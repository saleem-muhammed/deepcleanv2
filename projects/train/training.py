import h5py
import os

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


# Set the DeepCelanCLI
class DeepCleanCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--verbose", type=bool, default=False)

        parser.add_optimizer_args(torch.optim.Adam)
        # parser.add_lr_scheduler_args(torch.optim.lr_scheduler.OneCycleLR)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.StepLR)

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
        # parser.link_arguments(
        #     "data.steps_per_epoch",
        #     "lr_scheduler.steps_per_epoch",
        #     apply_on="instantiate",
        # )
        # parser.link_arguments("optimizer.lr", "lr_scheduler.max_lr")
        # parser.link_arguments("trainer.max_epochs", "lr_scheduler.epochs")
        parser.link_arguments(
            "optimizer.lr",
            "optimizer.lr",
            # apply_on="instantiate",
        )
        parser.link_arguments(
            "optimizer.weight_decay",
            "optimizer.weight_decay",
            # apply_on="instantiate",
        )
        parser.link_arguments(
            "lr_scheduler.step_size",
            "lr_scheduler.step_size",
            # apply_on="instantiate",
        )
        parser.link_arguments(
            "lr_scheduler.gamma",
            "lr_scheduler.gamma",
            # apply_on="instantiate",
        )

def main(args=None):
    tcli = DeepCleanCLI(
        model_class=dc_model,
        datamodule_class=dc_dataset,
        seed_everything_default=23984,
        run=False,
        parser_kwargs={"default_env": True},
        save_config_kwargs={"overwrite": True},
        args=args,
    )

    print(tcli.trainer.logger.log_dir)
    log_dir = tcli.trainer.logger.log_dir or tcli.trainer.logger.save_dir
    if not log_dir.startswith("s3://"):
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "train.log")
        configure_logging(log_file)
    else:
        configure_logging()
    # tcli.trainer.fit(tcli.model, tcli.datamodule)

    tcli.trainer.fit(tcli.model, tcli.datamodule)
    if tcli.datamodule.hparams.test_duration > 0:
        tcli.trainer.test(tcli.model, tcli.datamodule)


if __name__ == "__main__":
    main()
