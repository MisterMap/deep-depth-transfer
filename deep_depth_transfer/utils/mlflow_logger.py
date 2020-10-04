import pytorch_lightning as pl
import pytorch_lightning.loggers
import pytorch_lightning.utilities
import os


class MLFlowLogger(pl.loggers.MLFlowLogger):
    @pl.utilities.rank_zero_only
    def log_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        if not os.path.isdir("tmp"):
            os.mkdir("tmp")
        path = f"tmp/{tag}_{global_step:04d}.png"
        figure.savefig(path)
        self.experiment.log_artifact(self.run_id, path)
