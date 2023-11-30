import law
import luigi

from deepclean.base import DeepCleanTask


class Train(DeepCleanTask):
    train_config = luigi.Parameter(
        default="/opt/deepclean/projects/train/config.yaml"
    )
    data_fname = luigi.Parameter()
    output_dir = luigi.Parameter()
    use_wandb = luigi.BoolParameter()

    # TODO: should add offset parameters so that you
    # can train and test at arbitrary points inside
    # the data file.

    def make_name(self):
        problems = "_".join([i.value for i in self.cfg.problem])
        return "{}-{}-{}".format(
            self.cfg.ifo, problems, self.output().parent.basename
        )

    def configure_wandb(self, command: list[str]) -> None:
        command.append("--trainer.logger=WandbLogger")
        command.append("--trainer.logger.job_type=train")

        for key in ["name", "entity", "project", "group", "tags"]:
            value = getattr(self.cfg.wandb, key)
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
            self.train_config,
            "--data.fname",
            self.data_fname,
            "--data.channels",
            "[" + ",".join(channels) + "]",
            "--data.freq_low",
            str(self.cfg.freq_low),
            "--data.freq_high",
            str(self.cfg.freq_high),
        ]
        if self.use_wandb and not self.cfg.wandb.api_key:
            raise ValueError(
                "Can't run W&B experiment without specifying an API key. "
                "Please set the WANDB_API_KEY environment variable."
            )
        elif self.use_wandb:
            self.configure_wandb(command)
        command.append(f"--trainer.logger.save_dir={self.output_dir}")
        return command

    def output(self):
        output_dir = law.LocalDirectoryTarget(self.output_dir)
        return output_dir.child("model.pt", type="f")
