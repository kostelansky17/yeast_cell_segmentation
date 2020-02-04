import os

import imageio
import numpy as np
from PIL import Image, ImageSequence
import scipy.ndimage as ndi
from sklearn.model_selection import train_test_split

from settings import (
    BORDER_MASK_PATH,
    FILTER,
    PREPROCESSED_DATA_PATH,
    PREPROCESSED_MASK_PATH,
    RAW_DATA_PATH,
    RAW_MASK_PATH,
    SELECTED_U_NET_SETTINGS,
    TEST_SET_SIZE,
)

PREPROCESSING_INFO = [
    (RAW_DATA_PATH, PREPROCESSED_DATA_PATH, "data"),
    (RAW_MASK_PATH, PREPROCESSED_MASK_PATH, "ref"),
]


def create_borders_masks():
    """
    Preprocess data masks - loads masks describing positions of cells and changes
    them into masks decribing postions of boredes dividing them. The mask are 
    loaded from direcory PREPROCESSED_MASK_PATH and saved into BORDER_MASK_PATH.
    Bo
    
    All of these variables can be adjusted in 'settings.py'.
    """
    for mask_name in os.listdir(PREPROCESSED_MASK_PATH):
        mask_path = os.path.join(PREPROCESSED_MASK_PATH, mask_name)

        img = imageio.imread(mask_path, as_gray=True)

        threshold = (img > 0) * 255
        dilated = ndi.grey_dilation(threshold, size=(9, 9))
        dilated = ndi.grey_closing(dilated, size=(9, 9))
        out_img = dilated - threshold

        outfile = os.path.join(BORDER_MASK_PATH, mask_name)
        imageio.imwrite(outfile, out_img.astype(np.uint8))


def preprocess_raw_data():
    """
    Preprocess raw data - images in 'tif' format, each containing multiple frames. 
    The data are loaded from directory RAW_DATA_PATH and RAW_MASK_PATH respectively. 
    The preprocessed images are saved to the directories PREPROCESSED_MASK_PATH and 
    PREPROCESSED_REFERENCE_PATH.
    
    All of these variables can be adjusted in 'settings.py'.
    """
    for path_from, path_to, name in PREPROCESSING_INFO:
        for file in os.listdir(path_from):
            if file.endswith(".tif"):
                img = Image.open(path_from + file)
                save_image_frames_to(img, path_to, name + "_" + file.split(".")[0])


def save_image_frames_to(img, path_to, name):
    """
    Save each frame in image to specified location

    Parameters
    ----------
    img : PIL.Image
        Loaded image
    path_to : str
        Location of the directory in which will be the images saved
    name : str
        Prefix of the new image's name
    """
    for i, page in enumerate(ImageSequence.Iterator(img)):
        page.save(path_to + name + "_%d.png" % i)


def load_img_to_arr(shape, path):
    """
    Loads, reshape image from path and transform it to np.ndarray

    Parameters
    ----------
    shape : tuple of int
        Shape to reshape loaded image
    path : str
        Location of image

    Returns
    -------
    np.ndarray
        Loaded ans reshaped image transformed into np.ndarray
    """
    img = Image.open(path)
    height, width = img.size

    if height > 2000 and width > 2000:
        img = img.crop((500, 500, 1500, 1500))

    img = img.resize(shape)
    arr = np.array(img)
    return arr[:, :, np.newaxis]


def load_data(filter, X_shape, y_shape):
    """
    Loads, reshape image from path and transform it to np.ndarray

    Parameters
    ----------
    filter : callable
        Filter to used in data preprocessing
    X_shape : tuple of int, optional
        Shape to reshape loaded data images
    y_shape : tuple of int, optional
        Shape to reshape loaded mask images

    Returns
    -------
    tuple of np.ndarray
        Returns loaded X_train, X_test, y_train, y_test
    """
    X = []
    for file in sorted(os.listdir(PREPROCESSED_DATA_PATH)):
        if file.endswith(".png"):
            img = load_img_to_arr(X_shape, PREPROCESSED_DATA_PATH + file)
            if filter:
                img = filter(img)

            X.append(img)

    y = []
    for file in sorted(os.listdir(PREPROCESSED_MASK_PATH)):
        if file.endswith(".png"):
            img = load_img_to_arr(y_shape, PREPROCESSED_MASK_PATH + file)
            img[img > 0] = 1

            y.append(img)

    X, y = np.asarray(X), np.asarray(y)
    X_normalized = X / np.std(X, axis=0)
    return train_test_split(X_normalized, y, test_size=TEST_SET_SIZE, random_state=2)


def main():
    """
    Preprocess the training and test data. Their location
    can be set by adjusting variables:
        RAW_DATA_PATH,
        RAW_MASK_PATH,
        PREPROCESSED_DATA_PATH,
        PREPROCESSED_MASK_PATH,
    in the file settings.py
    """
    preprocess_raw_data()


if __name__ == "__main__":
    create_borders_masks()
