from ..cameras_calibration import CamerasCalibration
import numpy as np

class KittiEigenCamerasCalibration(CamerasCalibration):
    def __init__(self, final_size, original_size, device):
        height, width = final_size
        original_height, original_width = original_size
        scale = min(original_height / height, original_width / width)
        origin_focal = 707.0912
        original_cx = 601.8873
        original_cy = 183.1104
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
        camera_baseline = 0.54
        super().__init__(camera_baseline, left_camera_matrix, right_camera_matrix, device)

