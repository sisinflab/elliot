import numpy as np


class VisualLoader:

    def process_visual_features(self, data):
        self._f_feature = data.config.data_paths.feature_data
        self._emb_image = np.load(self._f_feature)
        self._num_image_feature = self._emb_image.shape[1]
        self._emb_image = self._emb_image / np.max(np.abs(self._emb_image))
