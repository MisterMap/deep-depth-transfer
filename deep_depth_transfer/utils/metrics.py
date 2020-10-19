import torch
import torch.nn.functional


# noinspection PyMethodMayBeStatic
class DepthMetric:
    def __init__(self):
        self._min_value = 1e-7
        self._metrics = [
            'RMSE', 'tRMSE', 'MAE', 'tMAE', 'RMSElog', 'SRD', 'ARD', 'SIlog', 'delta1', 'delta2',
            'delta3', 'PSNR', 'rPSNR'
        ]

    def threshold(self, y1, y2, threshold=1.25):
        ratio1 = y1 / y2
        ratio2 = y2 / y1
        ratio1: torch.Tensor
        max_ratio = torch.where(ratio1 > ratio2, ratio1, ratio2)
        result = torch.where(max_ratio < threshold, torch.tensor(1., device=y1.device),
                             torch.tensor(0., device=y1.device))
        return torch.mean(result) * 100.

    def rmse(self, y1, y2):
        diff = y1 - y2
        return torch.sqrt(torch.mean(diff * diff))

    def trmse(self, y1, y2, threshold=5):
        diff = y1 - y2
        value1 = diff * diff
        value2 = torch.tensor(threshold * threshold, device=value1.device, dtype=torch.float32)
        minimum = torch.where(value1 > value2, value2, value1)
        return torch.sqrt(torch.mean(minimum))

    def rmse_log(self, y1, y2):
        return self.rmse(torch.log(y1), torch.log(y2))

    def ard(self, y1, y2):
        return torch.mean(torch.abs(y1 - y2) / y2) * 100

    def srd(self, y1, y2):
        return torch.mean((y1 - y2) ** 2 / y2)

    def mae(self, y1, y2):
        return torch.mean(torch.abs(y1 - y2))

    def tmae(self, y1, y2, threshold=5):
        value1 = torch.abs(y1 - y2)
        value2 = torch.tensor(threshold, device=value1.device, dtype=torch.float32)
        minimum = torch.where(value1 > value2, value2, value1)
        return torch.mean(minimum)

    def psnr(self, y1, y2):
        rmse = self.rmse(y1, y2)
        if rmse == 0:
            return 100
        return 20 * torch.log10(torch.max(torch.abs(y1 - y2)) / rmse)

    def rpsnr(self, y1, y2):
        srd = self.srd(y1, y2)
        if srd == 0:
            return 100
        return 20 * torch.log10(torch.max(torch.abs(y1 - y2) / y2) / torch.sqrt(srd))

    def silog(self, y1, y2):
        d = torch.log(y1) - torch.log(y2)
        return 100 * torch.sqrt(torch.mean(d ** 2) - (torch.mean(d)) ** 2)

    def get_mask(self, output, ground_truth):
        return torch.logical_and(ground_truth > 0, output > 0)

    def clip(self, output, ground_truth, clip_min, clip_max):
        output_clipped = torch.clamp(output, clip_min, clip_max)
        ground_truth_clipped = torch.clamp(ground_truth, clip_min, clip_max)
        return output_clipped, ground_truth_clipped

    def __call__(self, output, ground_truth, clip_min=0., clip_max=25.):
        output = torch.nn.functional.interpolate(output, ground_truth.size()[-2:])
        mask = self.get_mask(output, ground_truth)
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
            self.psnr(output, ground_truth),
            self.rpsnr(output, ground_truth),
        ]
        result = {}
        for key, value in zip(self._metrics, values):
            result[key] = value
        return result

    def get_header(self):
        max_length = 10
        metrics = [(' ' * (max_length - len(x))) + x for x in self._metrics]
        return metrics
