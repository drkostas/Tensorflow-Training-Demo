import os
import numpy as np
from typing import *
import pandas as pd
from PIL import Image

base_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'fairface')


def load_dataset(dataset: str, n_rows: int = -1, delimiter=',') -> Tuple[List[Image.Image], pd.DataFrame]:
    """
    Load a dataset from a CSV file
    """
    labels_df = load_labels_into_df(dataset, n_rows, delimiter)
    files_list = labels_df['file'].tolist()
    return load_images(files_list), labels_df


def load_labels_into_df(dataset: str, n_rows: int = -1, delimiter=',') -> pd.DataFrame:
    """
    Load a CSV file using Pandas
    """

    file_name = f'fairface_label_{dataset}.csv'
    file_path = os.path.join(base_path, file_name)
    return pd.read_csv(file_path, delimiter=delimiter, nrows=(n_rows if n_rows != -1 else None))


def load_images(files_list: List[str]) -> Generator[Image.Image, None, None]:
    """
    Load images from a list of files
    """
    for file_name in files_list:
        file_path = os.path.join(base_path, file_name)
        image_pil = Image.open(file_path)
        # image_np = np.array(image_pil)
        yield image_pil
