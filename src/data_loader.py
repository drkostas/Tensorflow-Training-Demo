import os
import numpy as np
from typing import *
import pandas as pd
from PIL import Image
import pickle

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'fairface')
model_path = os.path.join(os.path.dirname(__file__), '..', 'models')


def load_dataset(dataset: str, n_rows: int = -1, delimiter=',') -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load images and labels
    """
    labels_df = load_labels_into_df(dataset, n_rows, delimiter)
    files_list = labels_df['file'].tolist()
    return load_images(files_list), labels_df


def load_labels_into_df(dataset: str, n_rows: int = -1, delimiter=',') -> pd.DataFrame:
    """
    Load a CSV file using Pandas
    """

    file_name = f'fairface_label_{dataset}.csv'
    file_path = os.path.join(data_path, file_name)
    return pd.read_csv(file_path, delimiter=delimiter, nrows=(n_rows if n_rows != -1 else None))


def load_images(files_list: List[str]) -> np.ndarray:
    """
    Load images from a list of files
    """
    images_lst = []
    for file_name in files_list:
        file_path = os.path.join(data_path, file_name)
        image_pil = Image.open(file_path)
        image_np = np.array(image_pil)
        images_lst.append(image_np)
    images_np = np.array(images_lst)
    return images_np


def save_pickle(data, file_name: str, task_name: str, model_name: str,
                protocol=pickle.HIGHEST_PROTOCOL):
    file_path = os.path.join(model_path, f'{task_name}_task', f'model_{model_name}')
    os.makedirs(file_path, exist_ok=True)
    file_path = os.path.join(file_path, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol=protocol)


def load_pickle(file_name: str, task_name: str, model_name: str):
    file_path = os.path.join(model_path, f'{task_name}_task', f'model_{model_name}', file_name)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
