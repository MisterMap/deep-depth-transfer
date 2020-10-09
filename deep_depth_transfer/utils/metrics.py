import math

import numpy as np
from skimage.measure import compare_ssim


# noinspection PyMethodMayBeStatic
class DepthMetric:

    def __init__(self):
        self._min_value = 1e-7
        self._metrics = ['RMSE', 'tRMSE', 'MAE', 'tMAE', 'RMSElog', 'SRD', 'ARD', 'SIlog', 'delta1', 'delta2',
                         'delta3', 'SSIM', 'PSNR', 'rPSNR']

    def threshold(self, y1, y2, threshold=1.25):
        max_ratio = np.maximum(y1 / y2, y2 / y1)
        return np.mean(max_ratio < threshold, dtype=np.float64) * 100.

    def rmse(self, y1, y2):
        diff = y1 - y2
        return math.sqrt(np.mean(diff * diff, dtype=np.float64))

    def trmse(self, y1, y2, thr=5):
        diff = y1 - y2
        return math.sqrt(np.mean(np.minimum(diff * diff, thr * thr), dtype=np.float64))

    def rmse_log(self, y1, y2):
        return self.rmse(np.log(y1), np.log(y2))

    def ard(self, y1, y2):
        return np.mean(np.abs(y1 - y2) / y2, dtype=np.float64) * 100

    def srd(self, y1, y2):
        return np.mean((y1 - y2) ** 2 / y2, dtype=np.float64)

    def mae(self, y1, y2):
        return np.mean(np.abs(y1 - y2), dtype=np.float64)

    def tmae(self, y1, y2, thr=5):
        return np.mean(np.minimum(np.abs(y1 - y2), thr), dtype=np.float64)

    def ssim(self, y1, y2):
        return compare_ssim(y1, y2)

    def psnr(self, y1, y2):
        rmse = self.rmse(y1, y2)
        if rmse == 0:
            return 100
        return 20 * math.log10(np.amax(np.abs(y1 - y2)) / rmse)

    def rpsnr(self, y1, y2):
        srd = self.srd(y1, y2)
        if srd == 0:
            return 100
        return 20 * math.log10(np.amax(np.abs(y1 - y2) / y2) / math.sqrt(srd))

    def silog(self, y1, y2):
        d = np.log(y1) - np.log(y2)
        return 100 * math.sqrt(np.mean(d ** 2, dtype=np.float64) - (np.mean(d, dtype=np.float64)) ** 2)

    def get_eval_pos(self, output, ground_truth):
        return np.logical_and(ground_truth > 0, output > 0)

    def clip(self, output, ground_truth, clip_min, clip_max):
        output_clipped = np.clip(output, clip_min, clip_max)
        ground_truth_clipped = np.clip(ground_truth, clip_min, clip_max)
        return output_clipped, ground_truth_clipped

    def calculate_metrics(self, output, ground_truth, clip_min=0., clip_max=25.):
        mask = np.logical_and(ground_truth > 0, output > 0)
        output = output[mask]
        ground_truth = ground_truth[mask]

        output, ground_truth = self.clip(output, ground_truth, clip_min=clip_min, clip_max=clip_max)

        values = [
            self.rmse(output, ground_truth),
            self.trmse(output, ground_truth),
            self.mae(output, ground_truth),
            self.tmae(output, ground_truth),
            self.rmse_log(output, ground_truth),
            self.srd(output, ground_truth),
            self.ard(output, ground_truth),
            self.silog(output, ground_truth),
            self.threshold(output, ground_truth, threshold=1.25),
            self.threshold(output, ground_truth, threshold=1.25 ** 2),
            self.threshold(output, ground_truth, threshold=1.25 ** 3),
            self.ssim(output, ground_truth),
            self.psnr(output, ground_truth),
            self.rpsnr(output, ground_truth)
        ]
        result = {}
        for key, value in zip(self._metrics, values):
            result[key] = value
        return result

    def get_header(self):
        max_length = 10
        metrics = [(' ' * (max_length - len(x))) + x for x in self._metrics]
        return metrics
