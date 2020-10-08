from .mflow_handler import MlFlowHandler
from .metrics import Metric
import torch
import cv2
import numpy as np


class EvaluateMonoDataset():
    def __init__(self, model, dataset, mlflow_tags=None, enable_mlflow=True, mlflow_parameters = None,
                 mlflow_experiment_name="TUM-RGBD"):
        self.model = model
        self.m = Metric()
        self._enable_mlflow = enable_mlflow
        if mlflow_parameters is None:
            mlflow_parameters = {}
        if self._enable_mlflow:
            self._mlflow_handler = MlFlowHandler(experiment_name=mlflow_experiment_name,
                                                 mlflow_tags=mlflow_tags, mlflow_parameters=mlflow_parameters)
        self.dataset = dataset
    
    def evaluate(self):
        metrics_all = []
        for val_dict in self.dataset:
            self.model.eval()
            device = "cuda:0"
            with torch.no_grad():
                pred_depth = self.model.depth(val_dict["tensor"].to(device, dtype=torch.float))
            depth_image = pred_depth[0].detach().cpu().permute(1, 2, 0).numpy()[:, :, 0]
            metrics_all.append(self.m.calc_metrics(depth_image, cv2.resize(val_dict["groundtruth_depth"], (384, 128))))
        result = np.array(metrics_all).mean(axis=0)
        if self._enable_mlflow:
            self._mlflow_loader(result)    
        return list(zip(self.m.get_header(), result))
        
    def _mlflow_loader(self, metrics):
        mlflow_dict = dict(zip(self.m.metrics, metrics))
        self._mlflow_handler.start_callback(mlflow_dict)
        self._mlflow_handler.finish_callback()