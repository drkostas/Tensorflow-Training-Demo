import traceback
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, optimizers, losses, metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Conv2D, MaxPooling2D

from src import *


def get_args() -> argparse.Namespace:
    """Set-up the argument parser

    Returns:
        argparse.Namespace:
    """
    parser = argparse.ArgumentParser(
        description='Project 3 for the Deep Learning class (COSC 525). '
                    'Involves the development of a Convolutional Neural Network.',
        add_help=False)
    # Required Args
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-t', '--task', type=str, required=True,
                               choices=['age', 'gender', 'race'], help="The task to train on.")
    # Optional args
    optional_args = parser.add_argument_group('Optional Arguments')
    optional_args.add_argument("--n-rows", default=-1, type=int, required=False,
                               help="How many rows of the dataset to read.")
    optional_args.add_argument("-h", "--help", action="help", help="Show this help message and exit")

    return parser.parse_args()


def build_model(input_shape: Tuple[int, int, int], n_classes: int, lr: float = 0.001) -> Model:
    """ Build a feed-forward convolutional neural network"""
    model = Sequential()
    # Add the layers
    model.add(Dense(1024, input_shape=input_shape, activation='tanh'))
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    # Select the optimizer and the loss function
    opt = optimizers.SGD(learning_rate=lr)
    model.compile(loss=losses.CategoricalCrossentropy, optimizer=opt)
    return model


def main():
    """This is the main function of train.py

    Example:
        python train.py --dataset example1
    """

    # Initializing
    args = get_args()
    # Load the dataset
    images_src, all_labels_src = load_dataset(dataset='train', n_rows=args.n_rows)
    # Extract the labels for the desired task
    print("All tasks: ", list(all_labels_src.columns)[1:-1])
    labels_src = all_labels_src[args.task].values
    # Split the train set into train and validation
    images_train, images_val, \
        labels_train, labels_val = split_data(images_src, labels_src, val_perc=0.2)
    # Scale the data
    min_max_dict = min_max_scale(images_train)
    images_train, train_min, train_max = \
        min_max_dict['data'], min_max_dict['min'], min_max_dict['max']
    images_val = min_max_scale(images_val, min_max_dict['max'], min_max_dict['min'])['data']
    # Save the min and max values of the train set for later use
    del min_max_dict['data']  # Don't need this anymore
    save_pickle(data=min_max_dict, file_name='min_max_dict.pkl', task_name=args.task, model_name='1')

    encoded_Labels = one_hot_encoder(labels_train)

    # ------- Start of Code ------- #
    model = build_model([5, images_train.shape[1], images_train.shape[2]], np.unique(encoded_Labels).size)
    print(model.summary())


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(str(e) + '\n' + str(traceback.format_exc()))
        raise e
