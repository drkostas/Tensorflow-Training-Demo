import traceback
import argparse
import numpy as np
import tensorboard
import datetime
import tensorflow as tf
from tensorflow.keras import Model, optimizers, losses, metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
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
    required_args.add_argument('-t', '--task', type=int, required=True,
                               choices=[1, 2, 3, 4, 5], help="The task/model to train on.")
    required_args.add_argument('-a', '--attr', type=str, required=True,
                               choices=['age', 'gender', 'race'], help="The attribute to train on.")
    # Optional args
    optional_args = parser.add_argument_group('Optional Arguments')
    optional_args.add_argument("--tuning", action='store_true', required=False,
                               help="Whether to use the validation or training set for training.")
    optional_args.set_defaults(feature=False)
    optional_args.add_argument("--n-rows", default=-1, type=int, required=False,
                               help="How many rows of the dataset to read.")
    optional_args.add_argument("-h", "--help", action="help", help="Show this help message and exit")

    return parser.parse_args()


def build_model_Dense(input_shape: Tuple[int, int], n_classes: int, lr: float = 0.001) -> Model:
    """ Build a feed-forward Dense neural network"""
    model = Sequential()
    # Add the layers
    model.add(Dense(1024, input_shape=input_shape, activation='tanh'))
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    # Select the optimizer and the loss function
    opt = optimizers.SGD(learning_rate=lr)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt,metrics = ['accuracy'])
    return model

def build_model_task_2_conv(input_shape: Tuple[int, int], n_classes: int, lr: float = 0.001) -> Model:
    """ Build a feed-forward conv neural network"""
    model = Sequential()
    input_shape = list(input_shape)
    input_shape.append(1)
    # Add the layers
    model.add(Conv2D(filters= 40, kernel_size = 5 , activation='relu', input_shape = input_shape))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    # Select the optimizer and the loss function
    opt = optimizers.SGD(learning_rate=lr)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt,metrics = ['accuracy'])


    return model


# def build_model(input_shape: Tuple[int, int], n_classes: int, lr: float = 0.001) -> Model:
#     """ Build a feed-forward convolutional neural network"""
#     model = Sequential()
#     # Add the layers
#     model.add(Dense(1024, input_shape=input_shape, activation='tanh'))
#     model.add(Dense(512, activation='sigmoid'))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(n_classes, activation='softmax'))
#     # Select the optimizer and the loss function
#     opt = optimizers.SGD(learning_rate=lr)
#     model.compile(loss=losses.CategoricalCrossentropy(), optimizer=opt)
#     return model


def main():
    """This is the main function of train.py

    Example:
        python train.py --dataset example1
    """

    # --- Hyper parameters --- #
    epochs = 10
    batch_size = 32
    lr = 0.001
    validation_set_perc = 0.01  # Percentage of the train dataset to use for validation

    # --- Initializing --- #
    args = get_args()
    if args.task == 1:
        build_model = build_model_Dense
    elif args.task==2:
        build_model = build_model_task_2_conv
    else:
        raise ValueError("Task not implemented")
    # Create a validation set suffix if needed
    val_set_suffix = ''
    if args.tuning:
        val_set_suffix = '_valset'
    # Load the dataset
    images_src, all_labels_src = load_dataset(dataset='train', n_rows=args.n_rows)
    images_test, all_labels_test = load_dataset(dataset='val', n_rows=args.n_rows)
    # Extract the labels for the desired task
    print("All tasks: ", list(all_labels_src.columns)[1:-1])
    labels_src = all_labels_src[args.attr].values
    labels_test = all_labels_test[args.attr].values
    # Split the train set into train and validation
    images_train, images_val, \
        labels_train, labels_val = split_data(images_src, labels_src, val_perc=validation_set_perc)
    # Scale the data
    min_max_dict = min_max_scale(images_train)
    images_train, train_min, train_max = \
        min_max_dict['data'], min_max_dict['min'], min_max_dict['max']
    images_val = min_max_scale(images_val, min_max_dict['max'], min_max_dict['min'])['data']
    # Save the min and max values of the train set for later use
    del min_max_dict['data']  # Don't need this anymore
    save_pickle(data=min_max_dict, file_name=f'min_max_dict{val_set_suffix}.pkl',
                attr=args.attr, task=args.task)
    # One hot encode the labels
    encoded_train_labels = one_hot_encoder(labels_train)
    encoded_val_labels = one_hot_encoder(labels_val)
    encoded_test_labels = one_hot_encoder(labels_test)
    # Select the train or validation set
    if args.tuning:
        images_train = images_val
        encoded_train_labels = encoded_val_labels

    # ------- Start of Code ------- #
    # --- Training --- #
    if(args.task ==1):
        # Flatten the images
        images_train = np.array([image.flatten() for image in images_train])
    # Build the model
    model = build_model(input_shape=images_train.shape[1:],
                        n_classes=encoded_train_labels.shape[1],
                        lr=lr)
    print(model.summary())

    log_folder = "logs/fit/t-"+str(args.task)+"b-" + str(batch_size)+"/lr-"+str(lr)

    callbacks = [TensorBoard(log_dir=log_folder,
                             histogram_freq=1,
                             write_graph=True,
                             write_images=False,
                             update_freq='epoch',
                             profile_batch=2,
                             embeddings_freq=1)]


    # Train the model
    model.fit(images_train,
              encoded_train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_split= validation_set_perc,
              callbacks=callbacks)

    # Run "tensorboard --logdir logs/fit" in terminal and open http://localhost:6006/

    # --- Evaluation --- #
    # Flatten the images
    if(args.task ==1):
        images_test = np.array([image.flatten() for image in images_test])
    # Evaluate the model



    #model.evaluate(images_test, encoded_test_labels)


    # Save the model
    # If we want to save every few epochs:
    # https://stackoverflow.com/a/59069122/7043716
    model_name = f'model_{epochs}epochs_{batch_size}batchsize_{lr}lr'
    if args.n_rows != -1:
        model_name += f'_{args.n_rows}rows'
    model_name += f'{val_set_suffix}.h5'
    save_path = os.path.join(model_path, f'{args.attr}_attr', f'task_{args.task}', model_name)
    model.save(save_path)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(str(e) + '\n' + str(traceback.format_exc()))
        raise e
