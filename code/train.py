from datetime import datetime
import math

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

from preprocessing import load_data
from settings import BATCH_SIZE, EPOCHS, VALIDATION_SPLIT
from u_net import build_u_net


def step_decay(epoch):
    """
    Learning rate decay function. Decays the learning rate by one half every ten epochs

    Parameters
    ----------
    epoch: int
        Current training epoch
    """
    initial_l_rate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    l_rate = initial_l_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return l_rate


def train_model(settings, X_train, y_train, save=True):
    """
    Trains and saves model

    Parameters
    ----------
    settings : dict
        Settings of a model, see 'settings.py'
    X_train : np.ndarray
        Training data - features
    y_train : np.ndarray
        Training data - labels
    save : Boolean, optional
        Value describing the need to save the model

    Returns
    -------
    model : tensorflow.keras.Model
        The trained model
    """
    model = build_u_net(*settings["model_settings"])

    l_rate = LearningRateScheduler(step_decay)
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=20)

    model.compile(
        Adam(), loss="binary_crossentropy", metrics=["accuracy", "Precision", "Recall"]
    )
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[es, l_rate],
    )

    if save:
        model_name = genereate_model_name()
        model.save(model_name)

    if save:
        model_name = genereate_model_name()
        model.save(model_name)

    return model


def genereate_model_name():
    """
    Generates filename for trained model, e.g. "model_12-24-2019_17:00.h5"

    Returns
    -------
    str
        Model filename
    """
    current_time = datetime.now()
    date_time = current_time.strftime("%m-%d-%Y_%H:%M")

    return "model_" + date_time + ".h5"
