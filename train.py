import traceback
import argparse
import numpy as np
from typing import *
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Conv2D, MaxPooling2D

from src import load_dataset, split_data


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
    model.compile(loss='categorical_cross_entropy', optimizer=opt)
    return model


def main():
    """This is the main function of train.py

    Example:
        python train.py --dataset example1
    """

    # Initializing
    args = get_args()
    # Load the dataset
    images, all_labels = load_dataset(dataset='train', n_rows=args.n_rows)
    print("All tasks: ", all_labels.columns)
    labels = all_labels[args.task].values
    print(labels.shape)
    # print(images.shape)
    images_train, images_test, images_val, \
        labels_train, images_test, images_val = split_data(images, labels, test_perc=0.1, val_perc=0.1)
    print(labels_train.shape)
    print(images_test.shape)
    print(images_val.shape)

    # ------- Start of Code ------- #


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(str(e) + '\n' + str(traceback.format_exc()))
        raise e
