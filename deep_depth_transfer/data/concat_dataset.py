import torch.utils.data


class ConcatDataset(torch.utils.data.ConcatDataset):
    def set_transform(self, transform):
        for dataset in self.datasets:
            dataset.set_transform(transform)
