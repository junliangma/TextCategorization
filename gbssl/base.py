from sklearn.base             import ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel


class MRBinaryClassifierMixin(ClassifierMixin):
    """Mixin for manifold regularization classifiers."""

    def __init__(self, gamma_a=1.0, gamma_i=1.0, gamma_k=None, p=1):
        self.gamma_a          = gamma_a
        self.gamma_i          = gamma_i
        self.gamma_k          = gamma_k
        self.p                = p

    def predict(self, X, Z):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        y : array-like, shape = [n_samples]
            Predictions for input data.
        """

        return rbf_kernel(X, Z, gamma=self.gamma_k) @ self.dual_coef_


class LMRBinaryClassifierMixin(ClassifierMixin):
    """Mixin for linear manifold regularization classifiers."""

    def __init__(self, gamma_a=1.0, gamma_i=1.0, p=1):
        self.gamma_a          = gamma_a
        self.gamma_i          = gamma_i
        self.p                = p

    def predict(self, X, Z):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        y : array-like, shape = [n_samples]
            Predictions for input data.
        """

        return X @ self.coef_
