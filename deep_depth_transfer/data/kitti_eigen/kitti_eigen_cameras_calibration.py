from ..cameras_calibration import CamerasCalibration


class KittiEigenCamerasCalibration(CamerasCalibration):
    def __init__(self, final_size, original_size, device):
        original_focal_x = 535.4
        original_focal_y = 539.2
        original_cx = 320.1
        original_cy = 247.6
        camera_matrix = self.calculate_camera_matrix(final_size, original_size, original_focal_x, original_focal_y,
                                                     original_cx, original_cy)
        camera_baseline = 1.0
        super().__init__(camera_baseline, camera_matrix, camera_matrix, device)
