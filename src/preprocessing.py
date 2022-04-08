from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def train_test_split_data(data, train_percentage):
    """
    Function to split the data into training and testing data
    """
    train_data, test_data = train_test_split(data,
                                             train_size=train_percentage)
    return train_data, test_data


def min_max_scale(train_data):
    """
    Function to scale the data to a range of 0 to 1
    """
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(train_data)
    scaled_data = min_max_scaler.transform(train_data)
    return scaled_data

# One hot encoding function
