import pytorch_lightning as pl
import pytorch_lightning.loggers
import pytorch_lightning.utilities


class LoggerCollection(pl.loggers.LoggerCollection):
    @pl.utilities.rank_zero_only
    def log_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        for logger in self._logger_iterable:
            logger.log_figure(tag, figure, global_step, close, walltime)
