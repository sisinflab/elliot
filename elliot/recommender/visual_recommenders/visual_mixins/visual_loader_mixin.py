from PIL import Image
import os
import numpy as np


class VisualLoader:

    def process_visual_features(self, data):
        self._f_feature = data.config.data_config.side_information.feature_data
        self._emb_image = np.load(self._f_feature)
        self._num_image_feature = self._emb_image.shape[1]
        self._emb_image = self._emb_image / np.max(np.abs(self._emb_image))

    @staticmethod
    def load_images(data):
        images_list = os.listdir(data.config.data_paths.image_data)
        images_list.sort(key=lambda x: int(x.split(".")[0]))

        for index, image in enumerate(images_list):
            im = Image.open(data.config.data_paths.image_data + image)
            try:
                im.load()
            except ValueError:
                print(f'Image at path '
                      f'{data.config.data_paths.image_data + image} was not loaded correctly!')
            if im.mode != 'RGB':
                im = im.convert(mode='RGB')
            im = np.reshape((np.array(im.resize((224, 224))) - np.float32(127.5)) / np.float32(127.5), (1, 224, 224, 3))
            yield index, im
