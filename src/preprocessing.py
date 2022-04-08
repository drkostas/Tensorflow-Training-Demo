from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def split_data(images, labels, test_perc, val_perc):
    """
    Function to split the data into training and testing data
    """
    train_perc = 1.0-test_perc-val_perc
    images_train, images_test, \
        labels_train, labels_test = train_test_split(images, labels,
                                                     test_size=1-train_perc)
    images_val, images_test, \
        labels_val, labels_test = train_test_split(images_test, labels_test,
                                                   test_size=test_perc/(test_perc + val_perc))
    return images_train, images_test, images_val, labels_train, images_test, images_val


def min_max_scale(train_data):
    """
    Function to scale the data to a range of 0 to 1
    """
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(train_data)
    scaled_data = min_max_scaler.transform(train_data)
    return scaled_data

# One hot encoding function
