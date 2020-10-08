import pytorch_lightning as pl
import pytorch_lightning.loggers
import pytorch_lightning.utilities


class TensorBoardLogger(pl.loggers.TensorBoardLogger):
    @pl.utilities.rank_zero_only
    def log_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        self.experiment.add_figure(tag, figure, global_step, close, walltime)
