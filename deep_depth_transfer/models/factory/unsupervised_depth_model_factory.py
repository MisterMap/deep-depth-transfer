from .. import PoseNetResNet, DepthNetResNet, UnsupervisedDepthModel
from ... import ResultVisualizer
from ...criterion import UnsupervisedCriterion


class UnsupervisedDepthModelFactory(object):
    @staticmethod
    def make_model(params, cameras_calibration):
        pose_net = PoseNetResNet()
        depth_net = DepthNetResNet()
        criterion = UnsupervisedCriterion(cameras_calibration)

        result_visualizer = ResultVisualizer(cameras_calibration=cameras_calibration)

        model = UnsupervisedDepthModel(
            params,
            pose_net,
            depth_net,
            criterion,
            result_visualizer=result_visualizer,
        )
        return model
