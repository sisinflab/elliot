import torch

from elliot.recommender.base_trainer import Trainer, TraditionalTrainer, GeneralTrainer
from elliot.utils.enums import ModelType


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(data, config, params, model_class):
    match model_class.type:
        case ModelType.BASE:
            trainer = Trainer

        case ModelType.TRADITIONAL:
            trainer = TraditionalTrainer

        case ModelType.GENERAL:
            trainer = GeneralTrainer

        case _:
            raise ValueError(f"Unknown model type '{model_class.type}'")

    return trainer(data, config, params, model_class)
