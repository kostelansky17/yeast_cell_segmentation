from preprocessing import load_data
from settings import FILTER, SELECTED_U_NET_SETTINGS
from test import test_model
from train import train_model


def main():
    """
    Trains and saves the model. Then tests the performacne of the model 
    on the test data set.
    """
    X_train, X_test, y_train, y_test = load_data(
        FILTER, *SELECTED_U_NET_SETTINGS["data_settings"]
    )
    model = train_model(SELECTED_U_NET_SETTINGS, X_train, y_train)
    test_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
