from .multi_unsupervised_depth_model_factory import MultiUnsupervisedDepthModelFactory
from .scaled_unsupervised_depth_model_factory import ScaledUnsupervisedDepthModelFactory
from .unsupervised_depth_model_factory import UnsupervisedDepthModelFactory


class ModelFactory(object):
    @staticmethod
    def make_model(params, cameras_calibration):
        if params.model_name == "multi_depth":
            return MultiUnsupervisedDepthModelFactory().make_model(params, cameras_calibration)
        elif params.model_name == "scaled_depth":
            return ScaledUnsupervisedDepthModelFactory().make_model(params, cameras_calibration)
        elif params.model_name == "unsupervised_depth":
            return UnsupervisedDepthModelFactory().make_model(params, cameras_calibration)
        else:
            raise ValueError(f"Unknown model name {params.model_name}")
