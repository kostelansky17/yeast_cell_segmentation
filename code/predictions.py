import os
import numpy as np
from PIL import Image, ImageSequence
from skimage.filters import gaussian
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from tensorflow.keras.models import load_model

from filters import highpass_filter
from preprocessing import load_img_to_arr
from settings import FILTER, PATH_TO_MODEL, SELECTED_U_NET_SETTINGS

IMG_SHAPE = SELECTED_U_NET_SETTINGS["data_settings"][0]


def prediction(path_to_image):
    """
    Takes path to image "path_to_image" and calculates predictions for each pixel,
    if the size of the image is bigger than 2000x2000 pixels, 500px is automatically
    cropped from each side. The size of the output image is equal to the output shape 
    of a given model. 

    Parameters
    ----------
    path_to_image : str
        Path to an image

    Returns
    -------
    np.ndarray
        Probability predictions for each pixel transformed into np.ndarray
    """
    loaded_image = load_img_to_arr(IMG_SHAPE, path_to_image)
    model = load_model(PATH_TO_MODEL)

    if FILTER is not None:
        loaded_image = FILTER(loaded_image)

    input_image = loaded_image[np.newaxis, ...]
    prediction = model.predict(input_image.astype(float))

    output_image = prediction[0, :, :, 0]

    return output_image


def threshold(pred, param):
    """
    Takes the predicted image "pred", thresholds it with the determined
    param, returns binary image.

    Parameters
    ----------
    pred : np.array
        Prediction image
    param : float
        Threshold for the input image

    Returns
    -------
    np.ndarray
        Binary np.array
    """

    pred[pred >= param] = 1
    pred[pred < param] = 0

    return pred


def segment(th, min_distance=10):
    """
    Performs watershed segmentation on thresholded, i.e. binary image

    Author
    ----------
    Minder Matthias, https://github.com/mattminder/segmentation_quality_measure/

    Parameters
    ----------
    th : np.array
        Loaded image
    min_distance : int
        Seeds have to have minimal distance of min_distance

    Returns
    -------
    np.ndarray
        Np.array of a segemented image
    """
    # Defines the watershed topology to be used
    topology = lambda x: gaussian(-x, sigma=0.5)

    dtr = ndi.morphology.distance_transform_edt(th)
    if topology is None:
        topology = -dtr
    elif callable(topology):
        topology = topology(dtr)

    m = peak_local_max(-topology, min_distance, indices=False)
    m_lab = ndi.label(m)[0]
    wsh = watershed(topology, m_lab, mask=th)
    return wsh


def main():
    """
    FOR TESTING PURPOSES ONLY
    """
    PATH_TO_IMAGE = "data_augoustina_first_im_0.png"
    p = prediction(PATH_TO_IMAGE)
    th = threshold(p, 0.5)
    s = segment(th, 4)


if __name__ == "__main__":
    main()
