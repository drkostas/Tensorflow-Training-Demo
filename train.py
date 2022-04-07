import traceback
import argparse
import numpy as np
from src import NeuralNetwork, generateExample, getTensorExample
from typing import *


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
    required_args.add_argument('-d', '--dataset', required=True,
                               help="The datasets to train the network on. "
                                    "Options: [example1, example2, example3]")
    # Optional args
    optional_args = parser.add_argument_group('Optional Arguments')
    optional_args.add_argument("-h", "--help", action="help", help="Show this help message and exit")

    return parser.parse_args()


def main():
    """This is the main function of main.py

    Example:
        python main.py --dataset example1
    """

    # Initializing
    args = get_args()
    # Load the configurations
    dataset_type = args.dataset
    if dataset_type in ('example1', 'example2', 'example3'):
        example_num = int(dataset_type[-1])
        inputs, targets, layers = generateExample(example_num)
        getTensorExample(example_num)
    else:
        raise ValueError('Invalid dataset type')

    # ------- Start of Code ------- #


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(str(e) + '\n' + str(traceback.format_exc()))
        raise e
