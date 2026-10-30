class ReferenceModel:
    """Persistence benchmark, not a selected production model."""
    requires_training = False

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["last_price"].to_numpy(dtype=float)
