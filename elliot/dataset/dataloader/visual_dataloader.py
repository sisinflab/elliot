"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Daniele Malitesta, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, daniele.malitesta@poliba.it, claudio.pomo@poliba.it'

import os
import numpy as np
import typing as t
import pandas as pd
import scipy.sparse as sp
from ast import literal_eval
from PIL import Image
from types import SimpleNamespace
import concurrent.futures as c

from dataset.abstract_dataset import AbstractDataset
from splitter.base_splitter import Splitter
from prefiltering.standard_prefilters import PreFilter

"""
[(train_0,test_0)]
[([(train_0,val_0)],test_0)]
[data_0]

[([(train_0,val_0), (train_1,val_1), (train_2,val_2), (train_3,val_3), (train_4,val_4)],test_0),
([(train_0,val_0), (train_1,val_1), (train_2,val_2), (train_3,val_3), (train_4,val_4)],test_1),
([(train_0,val_0), (train_1,val_1), (train_2,val_2), (train_3,val_3), (train_4,val_4)],test_2),
([(train_0,val_0), (train_1,val_1), (train_2,val_2), (train_3,val_3), (train_4,val_4)],test_3),
([(train_0,val_0), (train_1,val_1), (train_2,val_2), (train_3,val_3), (train_4,val_4)],test_4)]

[[data_0,data_1,data_2,data_3,data_4],
[data_0,data_1,data_2,data_3,data_4],
[data_0,data_1,data_2,data_3,data_4],
[data_0,data_1,data_2,data_3,data_4],
[data_0,data_1,data_2,data_3,data_4]]

[[data_0],[data_1],[data_2]]

[[data_0,data_1,data_2]]

[[data_0,data_1,data_2],[data_0,data_1,data_2],[data_0,data_1,data_2]]
"""


class VisualLoader:
    """
    Load train and test dataset
    """

    def __init__(self, config, *args, **kwargs):
        """
        Constructor of DataSet
        :param path_train_data: relative path for train file
        :param path_test_data: relative path for test file
        """

        self.args = args
        self.kwargs = kwargs
        self.config = config
        self.column_names = ['userId', 'itemId', 'rating', 'timestamp']

        if config.data_config.strategy == "fixed":
            path_train_data = config.data_config.train_path
            path_val_data = getattr(config.data_config, "validation_path", None)
            path_test_data = config.data_config.test_path
            visual_feature_path = getattr(config.data_config.side_information, "visual_features", None)
            item_mapping_path = getattr(config.data_config.side_information, "item_mapping", None)
            size_tuple = getattr(config.data_config.side_information, "output_image_size", None)


            if visual_feature_path and item_mapping_path:
                feature_set = set(pd.read_csv(item_mapping_path, sep="\t", header=None)[0].unique().tolist())
            else:
                feature_set = {}

            images_src_folder = getattr(config.data_config.side_information, "images_src_folder", None)

            if images_src_folder:
                image_set = {int(path.split(".")[0]) for path in os.listdir(images_src_folder)}
            else:
                image_set = {}

            if feature_set and image_set:
                visual_set = feature_set and image_set
            elif feature_set:
                visual_set = feature_set
            elif image_set:
                visual_set = image_set
            else:
                visual_set = {}

            self.side_information_data = SimpleNamespace()

            self.train_dataframe, self.side_information_data.aligned_items = self.load_dataset_dataframe(path_train_data,
                                                                                                       "\t",
                                                                                                       visual_set)
            self.side_information_data.visual_feature_path = visual_feature_path
            self.side_information_data.item_mapping_path = item_mapping_path
            self.side_information_data.images_src_folder = images_src_folder
            self.side_information_data.size_tuple = size_tuple

            self.train_dataframe = self.check_timestamp(self.train_dataframe)

            print('{0} - Loaded'.format(path_train_data))

            self.test_dataframe = pd.read_csv(path_test_data, sep="\t", header=None, names=self.column_names)
            self.test_dataframe = self.check_timestamp(self.test_dataframe)

            if path_val_data:
                self.validation_dataframe = pd.read_csv(path_val_data, sep="\t", header=None, names=self.column_names)
                self.validation_dataframe = self.check_timestamp(self.validation_dataframe)

                self.tuple_list = [([(self.train_dataframe, self.validation_dataframe)], self.test_dataframe)]
            else:
                self.tuple_list = [(self.train_dataframe, self.test_dataframe)]

        elif config.data_config.strategy == "hierarchy":
            self.tuple_list = self.read_splitting(config.data_config.root_folder)

        elif config.data_config.strategy == "dataset":
            print("There will be the splitting")
            path_dataset = config.data_config.dataset_path

            visual_feature_path = getattr(config.data_config.side_information, "visual_features", None)
            item_mapping_path = getattr(config.data_config.side_information, "item_mapping", None)
            size_tuple = getattr(config.data_config.side_information, "output_image_size", None)


            if visual_feature_path and item_mapping_path:
                feature_set = set(pd.read_csv(item_mapping_path, sep="\t", header=None)[0].unique().tolist())
            else:
                feature_set = {}

            images_src_folder = getattr(config.data_config.side_information, "images_src_folder", None)

            if images_src_folder:
                image_set = {int(path.split(".")[0]) for path in os.listdir(images_src_folder)}
            else:
                image_set = {}

            if feature_set and image_set:
                visual_set = feature_set and image_set
            elif feature_set:
                visual_set = feature_set
            elif image_set:
                visual_set = image_set
            else:
                visual_set = {}

            self.side_information_data = SimpleNamespace()

            self.dataframe, self.side_information_data.aligned_items = self.load_dataset_dataframe(path_dataset,
                                                                                                       "\t",
                                                                                                       visual_set)
            self.side_information_data.visual_feature_path = visual_feature_path
            self.side_information_data.item_mapping_path = item_mapping_path
            self.side_information_data.images_src_folder = images_src_folder
            self.side_information_data.size_tuple = size_tuple

            self.dataframe = self.check_timestamp(self.dataframe)

            print('{0} - Loaded'.format(path_dataset))

            self.dataframe = PreFilter.filter(self.dataframe, self.config.prefiltering)

            splitter = Splitter(self.dataframe, self.config.splitting)
            self.tuple_list = splitter.process_splitting()

        else:
            raise Exception("Strategy option not recognized")

    def check_timestamp(self, d: pd.DataFrame) -> pd.DataFrame:
        if all(d["timestamp"].isna()):
            d = d.drop(columns=["timestamp"]).reset_index(drop=True)
        return d

    def read_splitting(self, folder_path):
        tuple_list = []
        for dirs in os.listdir(folder_path):
            for test_dir in dirs:
                test_ = pd.read_csv(f"{folder_path}{test_dir}/test.tsv", sep="\t")
                val_dirs = [f"{folder_path}{test_dir}/{val_dir}/" for val_dir in os.listdir(f"{folder_path}{test_dir}") if os.path.isdir(f"{folder_path}{test_dir}/{val_dir}")]
                val_list = []
                for val_dir in val_dirs:
                    train_ = pd.read_csv(f"{val_dir}/train.tsv", sep="\t")
                    val_ = pd.read_csv(f"{val_dir}/val.tsv", sep="\t")
                    val_list.append((train_, val_))
                if not val_list:
                    val_list = pd.read_csv(f"{folder_path}{test_dir}/train.tsv", sep="\t")
                tuple_list.append((val_list, test_))

        return tuple_list

    def generate_dataobjects(self) -> t.List[object]:
        data_list = []
        for train_val, test in self.tuple_list:
            # testset level
            if isinstance(train_val, list):
                # validation level
                val_list = []
                for train, val in train_val:
                    single_dataobject = VisualDataObject(self.config, (train,val,test), self.side_information_data, self.args, self.kwargs)
                    val_list.append(single_dataobject)
                data_list.append(val_list)
            else:
                single_dataobject = VisualDataObject(self.config, (train_val, test), self.side_information_data, self.args,
                                                              self.kwargs)
                data_list.append([single_dataobject])
        return data_list

    def load_dataset_dataframe(self, file_ratings,
                               separator='\t',
                               visual_feature_set=None,
                               column_names=['userId', 'itemId', 'rating', 'timestamp']
                               ):
        data = pd.read_csv(file_ratings, sep=separator, header=None, names=column_names)

        if visual_feature_set is not None:
            data = data[data['itemId'].isin(visual_feature_set)]
            visual_feature_set = set(data['itemId'].unique()) and visual_feature_set

        return data, visual_feature_set

    def reduce_dataset_by_item_list(self, ratings_file, items, separator = '\t'):
        column_names = ["userId", "itemId", "rating"]
        data = pd.read_csv(ratings_file, sep=separator, header=None, names=column_names)
        data = data[data[column_names[1]].isin(items)]
        return data


class VisualDataObject:
    """
    Load train and test dataset
    """

    def __init__(self, config, data_tuple, side_information_data, *args, **kwargs):
        self.config = config
        self.side_information_data = side_information_data
        self.args = args
        self.kwargs = kwargs
        self.train_dict = self.dataframe_to_dict(data_tuple[0])

        if self.side_information_data.visual_feature_path:
            self.visual_features = np.load(self.side_information_data.visual_feature_path)
            self.item_mapping = pd.read_csv(self.side_information_data.item_mapping_path, sep="\t", header=None)
            self.item_mapping = {i: j for i, j in zip(self.item_mapping[0], self.item_mapping[1])}

        if self.side_information_data.images_src_folder:
            self.output_image_size = literal_eval(
                self.side_information_data.size_tuple) if self.side_information_data.size_tuple else None
            self.image_dict = self.read_images(self.side_information_data.images_src_folder, self.side_information_data.aligned_items, self.output_image_size)

        self.users = list(self.train_dict.keys())
        self.num_users = len(self.users)
        self.items = list(self.side_information_data.aligned_items)
        self.num_items = len(self.items)

        self.private_users = {p: u for p, u in enumerate(self.users)}
        self.public_users = {v: k for k, v in self.private_users.items()}
        self.private_items = {p: i for p, i in enumerate(self.items)}
        self.public_items = {v: k for k, v in self.private_items.items()}
        self.public_features = {v: k for k, v in self.private_features.items()}
        self.transactions = sum(len(v) for v in self.train_dict.values())

        self.i_train_dict = {self.public_users[user]: {self.public_items[i]: v for i, v in items.items()}
                                for user, items in self.train_dict.items()}

        self.sp_i_train = self.build_sparse()
        self.sp_i_train_ratings = self.build_sparse_ratings()

        if len(data_tuple) == 2:
            self.test_dict = self.build_dict(data_tuple[1], self.users)
        else:
            self.val_dict = self.build_dict(data_tuple[1], self.users)
            self.test_dict = self.build_dict(data_tuple[2], self.users)

    def read_images(self, images_folder, image_set, size_tuple):

        image_dict = {}
        for path in os.listdir(images_folder):
            image_id = int(path.split(".")[0])
            if image_id in image_set:
                im_pos = Image.open(os.path.join(images_folder, path))
                try:
                    im_pos.load()

                    if im_pos.mode != 'RGB':
                        im_pos = im_pos.convert(mode='RGB')

                    if size_tuple:
                        im_pos = np.array(im_pos.resize(size_tuple)) / np.float32(255)

                    image_dict[image_id] = im_pos
                except ValueError:
                    print(f'Image at path {pos.numpy()}.jpg was not loaded correctly!')
        return image_dict

    def read_images_multiprocessing(self, images_folder, image_set, size_tuple):

        samples = []
        image_dict = {}
        paths = [file for file in os.listdir(images_folder)]

        with c.ProcessPoolExecutor() as executor:
            workers = os.cpu_count()
            for offset_start in range(0, len(paths), workers):
                offset_stop = min(offset_start+workers, len(paths))

                results = executor.map(self.read_single_image,
                                       [images_folder] * workers,
                                       [image_set] * workers,
                                       [size_tuple] * workers,
                                       paths[offset_start:offset_stop])
                for result in results:
                    samples += result

        [image_dict.update(dict_) for dict_ in samples]
        return image_dict

    @staticmethod
    def read_single_image(images_folder, image_set, size_tuple, image_path):
        image_id = int(image_path.split(".")[0])
        if image_id in image_set:
            im_pos = Image.open(os.path.join(images_folder, image_path))
            try:
                im_pos.load()

                if im_pos.mode != 'RGB':
                    im_pos = im_pos.convert(mode='RGB')

                if size_tuple:
                    im_pos = np.array(im_pos.resize(size_tuple)) / np.float32(255)

                return {image_id: im_pos}
            except ValueError:
                print(f'Image at path {pos.numpy()}.jpg was not loaded correctly!')


    def dataframe_to_dict(self, data):
        users = list(data['userId'].unique())

        "Conversion to Dictionary"
        ratings = {}
        for u in users:
            sel_ = data[data['userId'] == u]
            ratings[u] = dict(zip(sel_['itemId'], sel_['rating']))
        n_users = len(ratings.keys())
        n_items = len({k for a in ratings.values() for k in a.keys()})
        transactions = sum([len(a) for a in ratings.values()])
        sparsity = 1 - (transactions / (n_users * n_items))
        print()
        print("********** Statistics")
        print(f'Users:\t{n_users}')
        print(f'Items:\t{n_items}', )
        print(f'Transactions:\t{transactions}')
        print(f'Sparsity:\t{sparsity}')
        print("********** ")
        return ratings

    def build_dict(self, dataframe, users):
        ratings = {}
        for u in users:
            sel_ = dataframe[dataframe['userId'] == u]
            ratings[u] = dict(zip(sel_['itemId'], sel_['rating']))
        return ratings

    def build_sparse(self):

        rows_cols = [(u, i) for u, items in self.i_train_dict.items() for i in items.keys()]
        rows = [u for u, _ in rows_cols]
        cols = [i for _, i in rows_cols]
        data = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32',
                             shape=(len(self.users), len(self.items)))
        return data

    def build_sparse_ratings(self):
        rows_cols_ratings = [(u, i, r) for u, items in self.i_train_dict.items() for i, r in items.items()]
        rows = [u for u, _, _ in rows_cols_ratings]
        cols = [i for _, i, _ in rows_cols_ratings]
        ratings = [r for _, _, r in rows_cols_ratings]

        data = sp.csr_matrix((ratings, (rows, cols)), dtype='float32',
                             shape=(len(self.users), len(self.items)))

        return data

    def get_test(self):
        return self.test_dict

    def get_validation(self):
        return self.val_dict if hasattr(self, 'val_dict') else None


