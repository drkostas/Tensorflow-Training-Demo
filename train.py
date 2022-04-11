import traceback
import argparse
from functools import partial
from tensorflow.keras import Model, optimizers, losses, metrics
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Activation, \
    Conv2D, MaxPooling2D, Lambda, Input, Conv2DTranspose, Reshape
from tensorflow.keras.losses import mse
from tensorflow.keras.callbacks import TensorBoard
import keras_tuner as kt
from tensorflow.keras import backend as K
from sklearn import metrics
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


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """
    # Extract mean and log of variance
    z_mean, z_log_var = args
    # get batch size and length of vector (size of latent space)
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]

    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    # Return sampled number (need to raise var to correct power)
    return z_mean + K.exp(z_log_var) * epsilon


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
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])
    return model


def build_model_task_2_conv(input_shape: Tuple[int, int], n_classes: int, lr: float = 0.001) -> Model:
    """ Build a feed-forward conv neural network"""
    model = Sequential()
    # input_shape = list(input_shape)
    # input_shape.append(1)
    # Add the layers
    model.add(Conv2D(filters=40, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    # Select the optimizer and the loss function
    opt = optimizers.SGD(learning_rate=lr)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])
    return model


def build_model_task_3_conv(input_shape: Tuple[int, int], n_classes: int,
                            lr: float = 0.00032) -> Model:
    """ Build a custom conv neural network"""
    model = Sequential()
    # Add the layers
    model.add(Conv2D(filters=64, kernel_size=3,
                     activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Conv2D(filters=64, kernel_size=3,
                     activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Conv2D(filters=128, kernel_size=3,
                     activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Conv2D(filters=128, kernel_size=3,
                     activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Dense(175, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    # Select the optimizer and the loss function
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=opt, metrics=['accuracy', 'mse'])
    return model


def build_model_task_5_auto(input_shape: Tuple[int, int],
                            n_classes: int, lr: float = 0.001) -> Tuple[Model, Model]:
    """ Build an Auto Encoder"""
    # Build Encoder
    inputs = Input(shape=input_shape, name='encoder_input')
    cov_1 = Conv2D(filters=20, kernel_size=5, activation='relu')(inputs)
    cov_2 = Conv2D(filters=20, kernel_size=5, activation='relu')(cov_1)
    cov_3 = Conv2D(filters=20, kernel_size=5, activation='relu')(cov_2)
    cov_4 = Conv2D(filters=20, kernel_size=5, activation='relu')(cov_3)
    flat = Flatten()(cov_4)
    z_mean = Dense(15, name='z_mean')(flat)
    z_log_var = Dense(15, name='z_log_var')(flat)
    # use reparameterization trick to push the sampling out as input
    z = Lambda(sampling, name='z')([z_mean, z_log_var])
    # instantiate encoder model
    encoder = Model(inputs, [cov_1, cov_2, cov_3, cov_4, z_mean, z_log_var, z], name='encoder_output')
    # build decoder model
    latent_inputs = Input(shape=(15,), name='z_sampling')
    x_out = Dense(5120, activation='relu', name="decoder_hidden_layer")(latent_inputs)
    resh = Reshape((16, 16, 20))(x_out)
    covT_1 = Conv2DTranspose(filters=20, kernel_size=5, activation='relu')(resh)
    covT_2 = Conv2DTranspose(filters=20, kernel_size=5, activation='relu')(covT_1)
    covT_3 = Conv2DTranspose(filters=20, kernel_size=5, activation='relu')(covT_2)
    covT_4 = Conv2DTranspose(filters=1, kernel_size=5, activation='relu')(covT_3)
    decoder = Model(latent_inputs, [x_out, resh, covT_1, covT_2, covT_3, covT_4],
                    name='decoder_output')
    decoder.compile(optimizer='adam')  # TODO: it gives the following error:
    # ValueError: The model cannot be compiled because it has no loss to optimize.
    outputs = decoder(encoder(inputs)[6])
    model = Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = mse(inputs, outputs[5])
    reconstruction_loss *= 1
    reconstruction_loss = K.mean(reconstruction_loss)
    kl_loss = K.exp(z_log_var) + K.square(z_mean) - z_log_var - 1
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= 0.001
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    model.add_loss(vae_loss)
    model.compile(optimizer='adam')
    return model, decoder


def tune_model_task_3_conv(hp, input_shape: Tuple[int, int], n_classes: int,
                           lr: float = 0.001, max_conv_layers: int = 3) -> Model:
    """ Build a feed-forward conv neural network"""
    # Tuning Params
    hp_cnn_activation = [hp.Choice(f'cnn_activation_{i}', values=['relu'], default='relu')
                         for i in range(max_conv_layers)]  # Only relu for now
    hp_dense_activation = hp.Choice('dense_activation', values=['relu'], default='relu')  # Only relu
    hp_filters = [hp.Choice(f'num_filters_{i}', values=[32, 64, 128], default=32)
                  for i in range(max_conv_layers)]
    hp_dense_units = hp.Int('dense_units', min_value=100, max_value=200, step=25)
    hp_lr = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG', default=1e-3)
    model = Sequential()
    # Add the layers
    for i in range(1, hp.Int("num_layers", 2, max_conv_layers + 1)):
        model.add(Conv2D(filters=hp_filters[i - 1], kernel_size=3,
                         activation=hp_cnn_activation[i - 1], input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Dense(hp_dense_units, activation=hp_dense_activation))
    model.add(Dense(n_classes, activation='softmax'))
    # Select the optimizer and the loss function
    opt = optimizers.Adam(learning_rate=hp_lr)
    # opt = optimizers.SGD(learning_rate=hp_lr)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=opt, metrics=['accuracy', 'mse'])
    return model

def main():
    """This is the main function of train.py

        Run "tensorboard --logdir logs/fit" in terminal and open http://localhost:6006/
    """
    # ---------------------- Hyperparameters ---------------------- #
    epochs = 1
    batch_size = 32
    tuning_image_num = 5000  # TODO: I'm already doing that in the data loader (you could use the --n-rows option and pass args.n_rows)
    tuning_epochs = 20
    # lr = 0.001  # For tasks 1 and 2
    lr = 0.00032  # For task 3
    validation_set_perc = 0.01  # Percentage of the train dataset to use for validation
    max_conv_layers = 4  # Only for tuning

    # ---------------------- Initialize variables ---------------------- #
    args = get_args()
    callbacks = []
    # Create a validation set suffix if needed
    val_set_suffix = ''
    if args.tuning:
        val_set_suffix = '_valset'
    log_folder = f"logs/fit{val_set_suffix}/t-" + str(args.task) + \
                 "/a-" + args.attr + \
                 "/b-" + str(batch_size) + \
                 "/lr-" + str(lr)
    if args.task == 1:
        build_model = build_model_Dense
    elif args.task == 2:
        build_model = build_model_task_2_conv
    elif args.task == 3:
        if args.tuning:
            build_model = tune_model_task_3_conv
        else:
            build_model = build_model_task_3_conv
    elif args.task == 5:
        build_model = build_model_task_5_auto
    else:
        raise ValueError("Task not implemented")

    # ---------------------- Load and prepare Dataset ---------------------- #
    # Load the dataset
    images_train, all_labels_src = load_dataset(dataset='train', n_rows=args.n_rows)
    images_test, all_labels_test = load_dataset(dataset='val', n_rows=args.n_rows)
    # Extract the labels for the desired task
    labels_train = all_labels_src[args.attr].values
    labels_test = all_labels_test[args.attr].values
    # Scale the data
    min_max_dict = min_max_scale(images_train)
    images_train, train_min, train_max = \
        min_max_dict['data'], min_max_dict['min'], min_max_dict['max']
    # Save the min and max values of the train set for later use
    del min_max_dict['data']  # Don't need this anymore
    save_pickle(data=min_max_dict, file_name=f'min_max_dict{val_set_suffix}.pkl',
                attr=args.attr, task=args.task)
    # One hot encode the labels
    encoded_train_labels = one_hot_encoder(labels_train)
    encoded_test_labels = one_hot_encoder(labels_test)

    # ---------------------- Build the Model ---------------------- #
    # Prepare images for training
    if args.task == 1:
        # Flatten the images
        images_train = np.array([image.flatten() for image in images_train])
    elif args.task in (2, 3, 4):
        images_train = images_train.reshape(*images_train.shape, 1)
    elif args.task == 5:
        images_train = images_train.reshape(*images_train.shape, 1)
        encoded_train_labels = images_train

    # Training/Tuning
    if not args.tuning:
        model = build_model(input_shape=images_train.shape[1:],
                            n_classes=encoded_train_labels.shape[1],
                            lr=lr)
        if args.task == 5:
            model, decoder = model
        print(model.summary())
    else:
        print("####### Tuning #######")
        build_model = partial(build_model, input_shape=images_train.shape[1:],
                              n_classes=encoded_train_labels.shape[1],
                              lr=lr, max_conv_layers=max_conv_layers)
        model = kt.Hyperband(build_model,
                             objective='val_accuracy',
                             factor=3,
                             directory=os.path.join(model_path,
                                                    f'{args.attr}_attr',
                                                    f'task_{args.task}'),
                             project_name=f'tuning_{epochs}epochs_{batch_size}batchsize_{lr}lr_max_conv_layers{max_conv_layers}')
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        callbacks.append(stop_early)
        model.search(images_train,
                     encoded_train_labels,
                     epochs=tuning_epochs,
                     batch_size=batch_size,
                     validation_split=validation_set_perc,
                     callbacks=callbacks)
        # Get the optimal hyperparameters
        # best_hps = model.get_best_hyperparameters(num_trials=1)[0]
        print("Best Model:")
        print(model.results_summary())
        print(model.search_space_summary())
        print("####### Tuning Done #######")
        return

    # ---------------------- Fit the Model ---------------------- #
    callbacks.append(TensorBoard(log_dir=log_folder,
                                 histogram_freq=1,
                                 write_graph=True,
                                 write_images=False,
                                 update_freq='epoch',
                                 profile_batch=2,
                                 embeddings_freq=1))
    model.fit(images_train,
              encoded_train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=validation_set_perc,
              callbacks=callbacks)

    file_writer = tf.summary.create_file_writer(log_folder)
    # Create Confusion Matrix
    if args.task in (1, 2, 3, 4):
        class_names = np.unique(labels_train)
        predictions = model.predict(images_train)
        predictions = np.argmax(predictions, axis=1)
        cm = metrics.confusion_matrix(np.argmax(encoded_train_labels, axis=1), predictions)
        figure = plot_confusion_matrix(cm, class_names=class_names)
        cm_image = plot_to_image(figure)
        with file_writer.as_default():
            tf.summary.image("Confusion Matrix", cm_image,step=epochs)

    # Create Images from the Auto Encoded
    if args.task == 5:
        figure = visualize_encoder_results(model, images_train)
        with file_writer.as_default():
            tf.summary.image("Image->Image", plot_to_image(figure), step=0)
        # Figure for Random Input
        figure = visualize_random_input(decoder)
        with file_writer.as_default():
            tf.summary.image("Random->Image", plot_to_image(figure), step=0)

    # ---------------------- Evaluation ---------------------- #
    # Flatten the images
    if args.task == 1:
        images_test = np.array([image.flatten() for image in images_test])
    elif args.task in (2, 3):
        images_train = images_train.reshape(*images_train.shape, 1)
    elif args.task == 5:
        images_train = images_train.reshape(*images_train.shape, 1)
        encoded_train_labels = images_train
    # Evaluate the model
    # model.evaluate(images_test, encoded_test_labels)

    # ---------------------- Save Model ---------------------- #
    # If we want to save every few epochs:
    # https://stackoverflow.com/a/59069122/7043716
    model_name = f'model_{epochs}epochs_{batch_size}batch-size_{lr}lr'
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
