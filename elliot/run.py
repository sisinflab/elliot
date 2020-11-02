from types import SimpleNamespace

import recommender
import dataset
import evaluation
import importlib
from yaml import load, dump
from yaml import CLoader as Loader, CDumper as Dumper, FullLoader as FullLoader
from utils.folder import manage_directories

if __name__ == '__main__':
    config_file = open('./config/config.yml')
    config = load(config_file, Loader=FullLoader)

    config['experiment']['training_set'] = config['experiment']['training_set']\
        .format(config['experiment']['dataset'])
    config['experiment']['validation_set'] = config['experiment']['validation_set'] \
        .format(config['experiment']['dataset'])
    config['experiment']['test_set'] = config['experiment']['test_set'] \
        .format(config['experiment']['dataset'])
    config['experiment']['features'] = config['experiment']['features'] \
        .format(config['experiment']['dataset'])

    config['experiment']['recs'] = config['experiment']['recs'] \
        .format(config['experiment']['dataset'])
    config['experiment']['wights'] = config['experiment']['wights'] \
        .format(config['experiment']['dataset'])

    manage_directories(config['experiment']['recs'], config['experiment']['wights'])

    base = SimpleNamespace(
        path_train_data=config['experiment']['training_set'],
        path_validation_data=config['experiment']['validation_set'],
        path_test_data=config['experiment']['test_set'],
        path_feature_data=config['experiment']['features'],
        path_output_rec_result=config['experiment']['recs'],
        path_output_rec_weight=config['experiment']['wights'],
        dataset=config['experiment']['dataset'],
        top_k=config['experiment']['top_k'],
        metrics=config['experiment']['metrics'],
        relevance=config['experiment']['relevance'],
    )

    for key in config['experiment']['models']:
        model_class = getattr(importlib.import_module("recommender"), key)
        model = model_class(config=base, params=SimpleNamespace(**config['experiment']['models'][key]))
        model.train()
