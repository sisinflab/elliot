import numpy as np
import pickle
import os


def write_csv(df, filename):
    """
    Args:
        df: pandas dataframe to write
        filename (str): path to store the dataframe
    """
    df.to_csv(filename, index=False)


def save_obj(obj, name):
    """
    Store the object in a pkl file
    :param obj: python object to be stored
    :param name: file name (Not insert .pkl)
    :return:
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def save_np(npy, filename):
    """
    Store numpy to memory.
    Args:
        npy: numpy to save
        filename (str): filename
    """
    np.save(filename, npy)


def store_recommendation(recommendations, path=""):
    """
    Store recommendation list (top-k)
    :return:
    """

    with open(path, 'w') as out:
        for u, recs in recommendations.items():
            for i, value in recs:
                out.write(str(u) + '\t' + str(i) + '\t' + str(value) + '\n')
