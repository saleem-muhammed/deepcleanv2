import law
import luigi

from deepclean.base import DeepCleanTask
from deepclean.config import deepclean as Config
from deepclean.config import wandb


class Train(DeepCleanTask):
    data_fname = luigi.Parameter()
    output_dir = luigi.Parameter()
    wandb = luigi.BoolParameter()

    cfg = Config()

    def make_name(self):
        return f"{self.cfg.ifo}-{self.cfg.problem}-{self.output().basename}"

    def configure_wandb(self, command: list[str]) -> None:
        command.append("--trainer.logger=WandbLogger")
        command.append("--trainer.logger.job_type=train")

        config = wandb()
        for key in ["name", "entity", "project", "group", "tags"]:
            value = getattr(config, key)
            if not value and key == "name":
                value = self.make_name()
            if value and key != "tags":
                command.append(f"--trainer.logger.{key}={value}")
            elif value:
                for v in value.split(","):
                    command.append(f"--trainer.logger.{key}+={v}")
        return command

    @property
    def command(self) -> list[str]:
        channels = [self.strain_channel] + self.witnesses
        command = [
            self.python,
            "/opt/deepclean/projects/train/train",
            "--config",
            "/opt/deepclean/projects/train/config.yaml",
            "--data.fname",
            self.data_fname,
            "--data.channels",
            "[" + ",".join(channels) + "]",
            "--data.freq_low",
            str(self.cfg.freq_low),
            "--data.freq_high",
            str(self.cfg.freq_high),
        ]
        if self.wandb and not wandb().api_key:
            raise ValueError(
                "Can't run W&B experiment without specifying an API key."
            )
        elif self.wandb:
            self.configure_wandb(command)
        command.append("--trainer.logger.save_dir=" + self.output().path)
        return command

    def output(self):
        return law.LocalDirectoryTarget(self.output_dir)
