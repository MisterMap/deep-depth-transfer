from .. import PoseNetResNet, DepthNetResNet, MultiUnsupervisedDepthModel, MultiDepthNet
from ... import ResultVisualizer
from ...criterion import UnsupervisedCriterion


class MultiUnsupervisedDepthModelFactory(object):
    @staticmethod
    def make_model(params, cameras_calibration):
        pose_net = PoseNetResNet()
        if params.depth_down_sample == "net":
            depth_net = MultiDepthNet()
        else:
            depth_net = DepthNetResNet()
        criterion = UnsupervisedCriterion(cameras_calibration)

        result_visualizer = ResultVisualizer(cameras_calibration=cameras_calibration)

        inner_criterions = {}
        for level in params.levels:
            if params.depth_down_sample == "up":
                cameras_calibration = cameras_calibration
            else:
                cameras_calibration = cameras_calibration.calculate_scaled_cameras_calibration(
                    scale=2 ** (level + 1),
                    image_size=params.image_size
                )
            inner_criterions[level] = UnsupervisedCriterion(
                cameras_calibration,
                lambda_s=params.inner_lambda_s,
                smooth_loss=False,
                pose_loss=False
            )
        model = MultiUnsupervisedDepthModel(
            params,
            pose_net,
            depth_net,
            criterion,
            inner_criterions=inner_criterions,
            result_visualizer=result_visualizer,
        )
        return model
