import pytorch_lightning as pl


class DepthEvaluationModel(pl.LightningModule):
    def __init__(self, depth_net, metrics):
        super().__init__()
        self._depth_net = depth_net
        self._metrics = metrics

    def losses(self, batch):
        depth = self._depth_net.depth(batch["image"])
        return self._metrics(depth, batch["ground_truth_depth"])

    def test_step(self, batch, batch_index):
        result = pl.EvalResult()
        result.log_dict(self.losses(batch))
        return result
