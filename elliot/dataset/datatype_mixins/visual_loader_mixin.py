import numpy as np


class VisualLoader:

    def process_visual_features(self, data):
        self.f_feature = data.kwargs['visual_features']
        self.emb_image = np.load(self.f_feature)
        self.num_image_feature = self.emb_image.shape[1]
        self.emb_image = self.emb_image / np.max(np.abs(self.emb_image))
