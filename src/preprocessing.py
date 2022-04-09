import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def split_data(images, labels, val_perc):
    """
    Function to split the data into training and testing data
    """
    images_train, images_val, \
        labels_train, labels_val = train_test_split(images, labels, test_size=val_perc)
    return images_train, images_val, labels_train, labels_val


def min_max_scale(train_data: np.ndarray, test_data: np.ndarray, val_data: np.ndarray):
    """
    Function to scale the data to a range of 0 to 1
    """
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(train_data)
    scaled_train = min_max_scaler.transform(train_data)
    scaled_test = min_max_scaler.transform(test_data)
    scaled_val = min_max_scaler.transform(val_data)
    return scaled_train, scaled_test, scaled_val

# One hot encoding function
