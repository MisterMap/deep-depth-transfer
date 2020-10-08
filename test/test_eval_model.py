import unittest

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from deep_depth_transfer import UnDeepVO
from deep_depth_transfer.problems import DepthModelEvaluator
from deep_depth_transfer.data.supervised import DataTransformManager
from deep_depth_transfer.problems import VideoVisualizer


class TestEvalModel(unittest.TestCase):
    # def test_model_loading(self):
    #     path = "checkpoint.pth"
    #     model = UnDeepVO(resnet=True).to("cuda:0")
    #     checkpoint = torch.load(path, map_location='cpu')
    #     model.load_state_dict(checkpoint)
    #
    # def test_model_evaluator(self):
    #     path = "checkpoint.pth"
    #     model = UnDeepVO(resnet=True).to("cuda:0")
    #     checkpoint = torch.load(path, map_location='cpu')
    #     model.load_state_dict(checkpoint)
    #     evaluator = DepthModelEvaluator(model)
    #     print(evaluator.calculate_metrics())

    def test_model_video(self):
        path = "checkpoint.pth"
        model = UnDeepVO(resnet=True).to("cuda:0")
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint)

        visualiser = VideoVisualizer(model, 'test2.mp4', 'out_d.mp4', 'out_img.mp4')
        visualiser.render()


if __name__ == '__main__':
    unittest.main()
