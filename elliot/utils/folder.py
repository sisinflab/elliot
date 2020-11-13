"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import os
import shutil


def manage_directories(path_output_rec_result, path_output_rec_weight, path_output_rec_performance):
    if os.path.exists(os.path.dirname(path_output_rec_result)):
        shutil.rmtree(os.path.dirname(path_output_rec_result))
    os.makedirs(os.path.dirname(path_output_rec_result))

    if os.path.exists(os.path.dirname(path_output_rec_weight)):
        shutil.rmtree(os.path.dirname(path_output_rec_weight))
    os.makedirs(os.path.dirname(path_output_rec_weight))

    if os.path.exists(os.path.dirname(path_output_rec_performance)):
        shutil.rmtree(os.path.dirname(path_output_rec_performance))
    os.makedirs(os.path.dirname(path_output_rec_performance))
