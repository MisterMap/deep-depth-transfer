import numpy as np
import torch


class CamerasCalibration(object):
    def __init__(self, camera_baseline, left_camera_matrix, right_camera_matrix, device="cuda:0"):
        self.camera_baseline = camera_baseline
        self.left_camera_matrix = torch.from_numpy(left_camera_matrix).to(device)[None].float()
        self.right_camera_matrix = torch.from_numpy(right_camera_matrix).to(device)[None].float()
        self.focal_length = self.left_camera_matrix[0, 0, 0]
        self.transform_from_left_to_right = torch.tensor(((1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1)))[None].to(device).float()
        self.transform_from_left_to_right[0, 0, 3] = -self.camera_baseline

    @staticmethod
    def calculate_camera_matrix(final_size, original_size, original_focal_x, original_focal_y,
                                original_cx, original_cy):
        height, width = final_size
        original_height, original_width = original_size
        scale = min(original_height / height, original_width / width)
        focal_x = original_focal_x / scale
        focal_y = original_focal_y / scale
        original_delta_cx = original_cx - original_width / 2
        original_delta_cy = original_cy - original_height / 2
        cx = width / 2 + original_delta_cx / scale
        cy = height / 2 + original_delta_cy / scale
        camera_matrix = np.array([[focal_x, 0., cx],
                                     [0., focal_y, cy],
                                     [0., 0., 1.]])
        return camera_matrix

    def calculate_scaled_cameras_calibration(self, image_size, scale):
        original_camera_matrix = self.left_camera_matrix[0].cpu().detach().numpy()
        camera_matrix = CamerasCalibration.calculate_camera_matrix((image_size[0] // scale, image_size[1] // scale),
                                                                   (image_size[0], image_size[1]),
                                                                   original_camera_matrix[0, 0],
                                                                   original_camera_matrix[1, 1],
                                                                   original_camera_matrix[0, 2],
                                                                   original_camera_matrix[1, 2])
        cameras_calibration = CamerasCalibration(self.camera_baseline, camera_matrix,
                                                 camera_matrix)
        return cameras_calibration
