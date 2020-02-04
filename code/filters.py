from scipy import ndimage


def highpass_filter(image):
    """
    Applies highpass filter to an image

    Parameters
    ----------
    image : np.ndarray
        Image to process

    Returns
    -------
    gauss_highpass : np.ndarray
        Processed image
    """
    lowpass = ndimage.gaussian_filter(image, 3)
    gauss_highpass = image - lowpass

    return gauss_highpass
