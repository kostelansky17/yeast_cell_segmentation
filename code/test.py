def test_model(model, X_test, y_test):
    """
    Test model on given test data

    Parameters
    ----------
    model : np.ndarray
        Model to test
    X_test : np.ndarray
        Test set data
    y_test : np.ndarray
        Test set labels
    """
    results = model.evaluate(X_test, y_test)
    print("Test loss, Test acc:", results)
