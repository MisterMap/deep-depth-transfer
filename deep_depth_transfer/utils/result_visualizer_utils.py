import kornia
import torch
import matplotlib.pyplot as plt


def generate_image(cameras_calibration, left_image, right_image, left_depth, right_depth):
    device = left_image.device
    with torch.no_grad():
        generated_left_image = kornia.warp_frame_depth(
            right_image[None],
            left_depth[None],
            cameras_calibration.transform_from_left_to_right.to(device),
            cameras_calibration.left_camera_matrix.to(device)
        )
        generated_right_image = kornia.warp_frame_depth(
            left_image[None],
            right_depth[None],
            torch.inverse(cameras_calibration.transform_from_left_to_right).to(device),
            cameras_calibration.left_camera_matrix.to(device)
        )
    return generated_left_image[0], generated_right_image[0]


def numpy_image(image):
    return image.detach().cpu().permute(1, 2, 0).numpy()


def numpy_depth(depth):
    return depth.detach().cpu().numpy()


def show_inner_spatial_loss(inner_images, inner_depths, cameras_calibration, **kwargs):
    left_image, right_image = inner_images[0], inner_images[2]
    left_depth, right_depth = inner_depths[0], inner_depths[2]
    left_generated, right_generated = generate_image(cameras_calibration, left_image[0], right_image[0],
                                                     left_depth[0], right_depth[0])
    delta = torch.mean(torch.abs(right_generated - right_image[0]), dim=0)
    figure, axes = plt.subplots(1, 1, **kwargs)
    axes.imshow(numpy_depth(delta))
    return figure
