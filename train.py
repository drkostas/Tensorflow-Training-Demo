import traceback
import argparse
import numpy as np
from typing import *
from src import load_dataset


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
    # Optional args
    optional_args = parser.add_argument_group('Optional Arguments')
    optional_args.add_argument("--n-rows", default=-1, type=int, required=False,
                               help="How many rows of the dataset to read.")
    optional_args.add_argument("-h", "--help", action="help", help="Show this help message and exit")

    return parser.parse_args()


def main():
    """This is the main function of train.py

    Example:
        python train.py --dataset example1
    """

    # Initializing
    args = get_args()
    # Load the dataset
    images, labels = load_dataset(dataset='train', n_rows=args.n_rows)
    first_img = next(images)
    print(first_img.format)
    print(first_img.mode)
    print(first_img.size)
    print(labels.count())

    # ------- Start of Code ------- #


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(str(e) + '\n' + str(traceback.format_exc()))
        raise e
