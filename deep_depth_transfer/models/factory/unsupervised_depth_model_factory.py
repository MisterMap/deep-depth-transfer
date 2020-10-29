from .. import PoseNetResNet, DepthNetResNet, UnsupervisedDepthModel
from ... import ResultVisualizer
from ...criterion import UnsupervisedCriterion, MonoUnsupervisedCriterion


class UnsupervisedDepthModelFactory(object):
    @staticmethod
    def make_model(params, cameras_calibration, mono, stereo):
        pose_net = PoseNetResNet()
        depth_net = DepthNetResNet()
        if stereo:
            criterion = UnsupervisedCriterion(cameras_calibration)
        else:
            criterion = MonoUnsupervisedCriterion(cameras_calibration)

        result_visualizer = ResultVisualizer(is_show_synthesized=stereo, cameras_calibration=cameras_calibration)

        model = UnsupervisedDepthModel(
            params,
            pose_net,
            depth_net,
            criterion,
            result_visualizer=result_visualizer,
            mono=mono,
            stereo=stereo,
            use_ground_truth_poses=params.use_poses
        )
        return model
