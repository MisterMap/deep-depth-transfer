import matplotlib.pyplot as plt
import numpy as np


class ResultVisualizer(object):
    def __init__(self, is_show_synthesized=False, max_depth=100., min_depth=0.1, dpi=150, batch_index=0):
        self._is_show_synthesized = is_show_synthesized
        self._max_depth = max_depth
        self._min_depth = min_depth
        self._dpi = dpi
        self.batch_index = batch_index

    def __call__(self, images, depths):
        left_current_image = images[0].cpu().permute(1, 2, 0).detach().numpy()
        left_current_depth = depths[0][0].cpu().detach().numpy()

        figure, axes = plt.subplots(2, 1, dpi=self._dpi)

        self.show_image(axes[0], left_current_image, "Left current image")
        self.show_depth(axes[1], left_current_depth, "Left current depth")
        figure.tight_layout()
        return figure

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
