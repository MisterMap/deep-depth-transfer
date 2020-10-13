import numpy as np
from ..cameras_calibration import CamerasCalibration


class SkoltechCamerasCalibrationFactory(object):
    @staticmethod
    def make_cameras_calibration(original_size, final_size, device):
        scale = 5.99
        height = 128
        width = 384
        origin_focal = (2.34694837e+03 + 2.35234114e+03 + 2.34706744e+03 + 2.35299521e+03) / 4
        original_cx = (1.17485473e+03 + 1.18452099e+03) / 2
        original_cy = (1.07139383e+03 + 1.01159783e+03) / 2
        original_height = 2048
        original_width = 2448
        focal = origin_focal / scale
        original_delta_cx = original_cx - original_width / 2
        original_delta_cy = original_cy - original_height / 2
        cx = width / 2 + original_delta_cx / scale
        cy = height / 2 + original_delta_cy / scale
        left_camera_matrix = np.array([[focal, 0., cx],
                                       [0., focal, cy],
                                       [0., 0., 1.]])
        right_camera_matrix = np.array([[focal, 0., cx],
                                        [0., focal, cy],
                                        [0., 0., 1.]])
        camera_baseline = 0.43
        return CamerasCalibration(camera_baseline, left_camera_matrix, right_camera_matrix, device)
