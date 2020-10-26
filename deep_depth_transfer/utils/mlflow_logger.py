import pytorch_lightning as pl
import pytorch_lightning.loggers
import pytorch_lightning.utilities
import os


class MLFlowLogger(pl.loggers.MLFlowLogger):
    @pl.utilities.rank_zero_only
    def log_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        try:
            if not os.path.isdir("tmp"):
                os.mkdir("tmp")
            path = f"tmp/{tag}_{global_step:04d}.png"
            figure.savefig(path)
            self.experiment.log_artifact(self.run_id, path)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as msg:
            print(msg)

    def log_metrics(self, **kwargs):
        try:
            super().log_metrics(**kwargs)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as msg:
            print(msg)

    def log_hyperparams(self, *args, **kwargs):
        try:
            super().log_hyperparams(*args, **kwargs)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as msg:
            print(msg)
