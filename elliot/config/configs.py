"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'


# DATASET
data_path = '../data/{0}/'
imagenet_classes_path = '../data/imagenet_classes.txt'
training_path = data_path + 'trainingset.tsv'
test_path = data_path + 'testset.tsv'
original = data_path + 'original/'
images_path = original + 'images/'
classes_path = original + 'classes.csv'
features_path = original + 'features.npy'

# RESULTS
weight_dir = '../results/rec_model_weight'
results_dir = '../results/rec_results'
