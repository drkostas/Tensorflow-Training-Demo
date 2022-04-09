import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import *


def split_data(images, labels, val_perc):
    """
    Function to split the data into training and testing data
    """
    images_train, images_val, \
        labels_train, labels_val = train_test_split(images, labels, test_size=val_perc)
    return images_train, images_val, labels_train, labels_val


def min_max_scale(data: np.ndarray, max_v: float = None, min_v: float = None) -> Dict:
    """
    Function to scale the data to a range of 0 to 1
    """
    return_dict = {}
    if max_v is None or min_v is None:
        max_v = np.max(data)
        min_v = np.min(data)
        return_dict['max'] = max_v
        return_dict['min'] = min_v
    return_dict['data'] = (data-min_v)/(max_v-min_v)
    return return_dict

# One hot encoding function
