class VideoDatasetAdapter(object):
    def __init__(self, kitty_dataset, index):
        self._dataset = kitty_dataset
        self._index = index
        self._img_size = self._dataset.get_rgb(0)[0].size[::-1]  # PIL Image size

    def get_image_size(self):
        return self._dataset.get_rgb(0)[0].size[::-1]

    def __getitem__(self, index):
        return self._dataset.get_rgb(index)[self._index]

    def __len__(self):
        return len(self._dataset.cam2_files)
