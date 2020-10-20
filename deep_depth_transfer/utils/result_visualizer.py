import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch


class ResultVisualizer(object):
    def __init__(self, is_show_synthesized=True, max_depth=100., min_depth=0.1, dpi=150, batch_index=0,
                 cameras_calibration=None):
        self._is_show_synthesized = is_show_synthesized
        self._cameras_calibration = cameras_calibration
        self._max_depth = max_depth
        self._min_depth = min_depth
        self._dpi = dpi
        self.batch_index = batch_index

    def generate_image(self, left_image, right_image, left_depth, right_depth):
        device = left_image.device
        with torch.no_grad():
            generated_left_image = kornia.warp_frame_depth(
                right_image[None],
                left_depth[None],
                self._cameras_calibration.transform_from_left_to_right.to(device),
                self._cameras_calibration.left_camera_matrix.to(device)
            )
            generated_right_image = kornia.warp_frame_depth(
                left_image[None],
                right_depth[None],
                torch.inverse(self._cameras_calibration.transform_from_left_to_right).to(device),
                self._cameras_calibration.left_camera_matrix.to(device)
            )
        return generated_left_image[0], generated_right_image[0]

    @staticmethod
    def numpy_image(image):
        return image.cpu().permute(1, 2, 0).detach().numpy()

    @staticmethod
    def numpy_depth(depth):
        return depth[0].cpu().detach().numpy()

    def show_generated_depth_figure(self, image, depth):
        figure, axes = plt.subplots(2, 1, dpi=self._dpi)
        self.show_image(axes[0], self.numpy_image(image), "Image")
        self.show_depth(axes[1], self.numpy_depth(depth), "Depth")
        figure.tight_layout()
        return figure

    def show_synthesized_figure(self, left_image, right_image, left_depth, right_depth):
        generated_left_image, generated_right_image = self.generate_image(left_image, right_image,
                                                                          left_depth, right_depth)

        figure, axes = plt.subplots(3, 2, dpi=self._dpi)
        self.show_image(axes[0][0], self.numpy_image(left_image), "Left image")
        self.show_image(axes[0][1], self.numpy_image(right_image), "Right image")
        self.show_depth(axes[1][0], self.numpy_depth(left_depth), "Left depth")
        self.show_depth(axes[1][1], self.numpy_depth(right_depth), "Right depth")
        self.show_image(axes[2][0], self.numpy_image(generated_left_image), "Generated left image")
        self.show_image(axes[2][1], self.numpy_image(generated_right_image), "Generated right image")
        figure.tight_layout()
        return figure

    def __call__(self, images, depths):
        if self._is_show_synthesized:
            return self.show_synthesized_figure(images[0], images[2], depths[0], depths[2])
        else:
            return self.show_generated_depth_figure(images[0], depths[0])

    @staticmethod
    def show_image(axis, image, caption="None"):
        axis.imshow(np.clip(image, 0, 1))
        axis.set_title(caption)
        axis.set_xticks([])
        axis.set_yticks([])

    def show_depth(self, axis, image, caption="None"):
        axis.imshow(self._min_depth / np.clip(image, self._min_depth, self._max_depth), cmap="inferno")
        axis.set_title(caption)
        axis.set_xticks([])
        axis.set_yticks([])
