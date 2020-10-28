from .. import PoseNetResNet, DepthNetResNet, ScaledUnsupervisedDepthModel
from ... import ResultVisualizer
from ...criterion import UnsupervisedCriterion


class ScaledUnsupervisedDepthModelFactory(object):
    @staticmethod
    def make_model(params, cameras_calibration):
        pose_net = PoseNetResNet()
        depth_net = DepthNetResNet()
        criterion = UnsupervisedCriterion(cameras_calibration)

        result_visualizer = ResultVisualizer(cameras_calibration=cameras_calibration)

        model = ScaledUnsupervisedDepthModel(
            params,
            pose_net,
            depth_net,
            criterion,
            result_visualizer=result_visualizer,
        )
        return model
