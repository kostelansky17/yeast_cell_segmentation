from scipy.ndimage import sobel

from filters import highpass_filter


# model training hyper-parameters
BATCH_SIZE = 1
EPOCHS = 200
VALIDATION_SPLIT = 0.2

# U-net architecture settings
BIG_U_NET_SETTINGS = {
    "model_settings": (1148, 32, 5),
    "data_settings": [(1148, 1148), (772, 772)],
}

MEDIUM_U_NET_SETTINGS = {
    "model_settings": (572, 64, 4),
    "data_settings": [(572, 572), (388, 388)],
}

SMALL_U_NET_SETTINGS = {
    "model_settings": (284, 64, 3),
    "data_settings": [(284, 284), (196, 196)],
}

# Selected architecture for training
SELECTED_U_NET_SETTINGS = SMALL_U_NET_SETTINGS

# Share of the test data in the dataset, value in range [0.0, 1.0]
TEST_SET_SIZE = 0.1

# Convolutional layer settings
CONV_LAYER_SETTINGS = {
    "kernel_size": (3, 3),
    "strides": 1,
    "activation": "elu",
    "kernel_initializer": "he_normal",
}

# filters available for preprocessing (highpass_filter/sobel/None)
FILTER = highpass_filter

# path to model to be used in predictions.py
PATH_TO_MODEL = "project_2/models/small_none.h5"

# path to folder with masks preprocessed to detect borders
BORDER_MASK_PATH = "project_2/code/data/border_mask/"

# paths to folders with raw data
RAW_DATA_PATH = "project_2/code/data/raw_data/"
RAW_MASK_PATH = "project_2/code/data/raw_mask/"

# paths to folders with preprocessed data (or/and where to save preprocessed data)
PREPROCESSED_DATA_PATH = "project_2/code/data/preprocessed_data/"
PREPROCESSED_MASK_PATH = "project_2/code/data/preprocessed_mask/"
