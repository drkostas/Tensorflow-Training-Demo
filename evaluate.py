import traceback
import argparse
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
    optional_args.add_argument('-o', '--attr2', type=str, required=False,
                               choices=['age', 'gender', 'race'], help="The Second attribute to train on. Only for Task 4.")
    return parser.parse_args()


def main():
    """This is the main function of train.py

        Run "tensorboard --logdir logs/fit" in terminal and open http://localhost:6006/
    """
    args = get_args()
    # ---------------------- Hyperparameters ---------------------- #

    if args.task == 1:
        epochs = 60
        saved_epoch = 60
        saved_lr = 0.001
        saved_batch_size = 128
    elif args.task == 2:
        epochs = 45
        saved_epoch = 45
        saved_lr = 0.001
        saved_batch_size = 128
    elif args.task == 3:
        epochs = 20
        saved_epoch = 8
        saved_lr = 0.00032  # For task 3
        saved_batch_size = 128
    else:
        epochs = 70
        saved_epoch = 70
        saved_lr = 0.00032
        saved_batch_size = 32

    # ---------------------- Initialize variables ---------------------- #
    # Create a validation set suffix if needed
    val_set_suffix = ''
    if args.tuning:
        val_set_suffix = '_valset'

    model_name = f'model_{epochs}epochs_{saved_batch_size}batch-size_{saved_lr}lr'
    if args.n_rows != -1:
        model_name += f'_{args.n_rows}rows'
    model_name += f'{val_set_suffix}.h5'
    save_dir_path = os.path.join(model_path, f'{args.attr}_attr', f'task_{args.task}')
    if(args.task == 4):
        save_filename = os.path.join(save_dir_path, model_name)
    else:
        save_filename = os.path.join(save_dir_path, model_name[:-3] + f'_epoch{saved_epoch:02d}.ckpt')


    log_folder = "logs/fit/t-" + str(args.task) + \
                 "/a-" + args.attr + \
                 "/b-" + str(saved_batch_size) + \
                 "/lr-" + str(saved_lr)


    # ---------------------- Load and prepare Dataset ---------------------- #
    # Load the dataset
    images_test, all_labels_test = load_dataset(dataset='val', n_rows=args.n_rows)
    # Extract the labels for the desired task
    labels_test = all_labels_test[args.attr].values

    if args.task == 4:
        labels_test_2 = all_labels_test[args.attr2].values
    # Scale the data
    min_max_dict = load_pickle(file_name=f'min_max_dict{val_set_suffix}.pkl',
                               attr=args.attr, task=args.task)
    min_max_dict = min_max_scale(images_test, max_v=min_max_dict['max'], min_v=min_max_dict['min'])
    images_test = min_max_dict['data']
    # One hot encode the labels
    encoded_test_labels = one_hot_encoder(labels_test)
    if args.task == 4:
        encoded_test_labels_2 = one_hot_encoder(labels_test_2)
    # ---------------------- Build the Model ---------------------- #
    # Prepare images for testing
    if args.task == 1:
        # Flatten the images
        images_test = np.array([image.flatten() for image in images_test])
    elif args.task in (2, 3):
        images_test = images_test.reshape(*images_test.shape, 1)
    elif args.task ==4:
        images_test = images_test.reshape(*images_test.shape, 1)
        encoded_test_labels = [encoded_test_labels,encoded_test_labels_2]
    elif args.task == 5:
        images_test = images_test.reshape(*images_test.shape, 1)
        encoded_test_labels = images_test




    # Load Model
    model = tf.keras.models.load_model(save_filename)
    print(model.summary())



    # ---------------------- Evaluation ---------------------- #
    # Evaluate the model
    model.evaluate(images_test, encoded_test_labels)
    model.predict(images_test)




    file_writer = tf.summary.create_file_writer(log_folder)


    if args.task in (1, 2, 3):
        class_names = np.unique(labels_test)
        predictions = model.predict(images_test)
        predictions = np.argmax(predictions, axis=1)
        cm = metrics.confusion_matrix(np.argmax(encoded_test_labels, axis=1), predictions)
        figure = plot_confusion_matrix(cm, class_names=class_names)
        cm_image = plot_to_image(figure)
        with file_writer.as_default():
            tf.summary.image("Evaluation Confusion Matrix For Task "+str(args.task)+" "+str(args.attr), cm_image,step=epochs)

    if args.task == 4:
        class_names = np.unique(labels_test)
        predictions = model.predict(images_test)
        predictions_1 = np.argmax(predictions[0], axis=1)
        cm = metrics.confusion_matrix(np.argmax(encoded_test_labels[0], axis=1), predictions_1)
        figure = plot_confusion_matrix(cm, class_names=class_names)
        cm_image = plot_to_image(figure)
        with file_writer.as_default():
            tf.summary.image("Evaluation Confusion Matrix For Task "+str(args.task)+" "+str(args.attr), cm_image,step=epochs)

        class_names = np.unique(labels_test_2)
        predictions_2 = np.argmax(predictions[1], axis=1)
        cm = metrics.confusion_matrix(np.argmax(encoded_test_labels[1], axis=1), predictions_2)
        figure = plot_confusion_matrix(cm, class_names=class_names)
        cm_image = plot_to_image(figure)
        with file_writer.as_default():
            tf.summary.image("Evaluation Confusion Matrix For Task "+str(args.task)+" "+str(args.attr2),
                             cm_image, step=epochs)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(str(e) + '\n' + str(traceback.format_exc()))
        raise e
