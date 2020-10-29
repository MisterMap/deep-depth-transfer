from .. import PoseNetResNet, DepthNetResNet, MultiUnsupervisedDepthModel, MultiDepthNet
from ... import ResultVisualizer
from ...criterion import UnsupervisedCriterion, MonoUnsupervisedCriterion


class MultiUnsupervisedDepthModelFactory(object):
    @staticmethod
    def make_model(params, cameras_calibration, mono, stereo):
        pose_net = PoseNetResNet()
        if params.depth_down_sample == "net" or params.depth_down_sample == "net_image":
            depth_net = MultiDepthNet()
        else:
            depth_net = DepthNetResNet()
        if stereo:
            criterion = UnsupervisedCriterion(cameras_calibration)
        else:
            criterion = MonoUnsupervisedCriterion(cameras_calibration)

        result_visualizer = ResultVisualizer(is_show_synthesized=stereo, cameras_calibration=cameras_calibration)

        inner_criterions = {}
        for level in params.levels:
            if params.depth_down_sample == "up":
                cameras_calibration = cameras_calibration
            else:
                cameras_calibration = cameras_calibration.calculate_scaled_cameras_calibration(
                    scale=2 ** (level + 1),
                    image_size=params.image_size
                )
            if stereo:
                inner_criterions[level] = UnsupervisedCriterion(
                    cameras_calibration,
                    lambda_s=params.inner_lambda_s,
                    smooth_loss=False,
                    pose_loss=False
                )
            else:
                inner_criterions[level] = MonoUnsupervisedCriterion(
                    cameras_calibration,
                    lambda_s=params.inner_lambda_s,
                    smooth_loss=False,
                )
        model = MultiUnsupervisedDepthModel(
            params,
            pose_net,
            depth_net,
            criterion,
            inner_criterions=inner_criterions,
            result_visualizer=result_visualizer,
            mono=mono,
            stereo=stereo,
            use_ground_truth_poses=params.use_poses
        )
        return model
