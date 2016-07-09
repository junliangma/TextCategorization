import numpy        as np
import scipy.sparse as sp
import scipy.linalg as LA

from .base import BaseMR, LinearMRClassifierMixin


class LinearLapRLSC(BaseMR, LinearMRClassifierMixin):
    """Linear Laplacian Regularized Least Squares Classifier.

    Parameters
    ----------
    gamma_a : float
        Regularization parameter.

    gamma_i : float
        Smoothness regularization parameter.

    sparsify : {'kNN', 'MkNN', 'epsilonNN'}
        Graph sparsification type.

    n_neighbors : integer > 0
        Number of neighbors for each sample.

    radius : float
        Radius of neighborhoods.

    reweight : {'binary', 'rbf'}
        Edge re-weighting type

    t : float
        Kernel coefficient.

    normed : boolian, default True
        If True, then compute normalized Laplacian.

    p : integer > 0
        Degree of the graph Laplacian.

    Attributes
    ----------
    X_ : array-like, shape = [n_samples, n_features], dtype = float64
        Training data.

    y_ : array-like, shape = [n_samples], dtype = float64
        Target values.

    A_ : array-like, shape = [n_samples, n_samples], dtype = float64
        Adjacency matrix.

    coef_ : array-like, shape = [n_features], dtype = float64
        Weight vector.

    References
    ----------
    Vikas Sindhwani, Partha Niyogi, Mikhail Belkin,
    "Linear Manifold Regularization for Large Scale Semi-supervised Learning",
    Proc. of 22nd ICML Workshop on Learning with Partially Classified Training data, 2005.
    """

    def __init__(
        self,            gamma_a = 1.0,  gamma_i     = 1.0, sparsify = 'kNN',
        reweight = True, t       = None, n_neighbors = 1,   radius   = 1.0,
        normed   = True, p       = 1
    ):
        super(LinearLapRLSC, self).__init__(
            gamma_a = gamma_a, gamma_i     = 0.0,         sparsify = sparsify, reweight = reweight,
            t       = t,       n_neighbors = n_neighbors, radius   = radius,   normed   = normed,
            p       = p
        )

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features], dtype = float64
            Training data.

        y : array-like, shape = [n_samples], dtype = float64
            Target values (unlabeled points are marked as 0).

        Returns
        -------
        self : Returns an instance of self.
        """

        labeled               = y != 0
        X_labeled             = X[labeled]
        y_labeled             = y[labeled]
        X_unlabeled           = X[-labeled]
        y_unlabeled           = y[-labeled]
        self.X_               = np.vstack((X_labeled, X_unlabeled))
        self.y_               = np.hstack((y_labeled, y_unlabeled))

        n_samples, n_features = self.X_.shape
        n_labeled_samples     = y_labeled.size
        n_classes             = np.unique(y_labeled).size
        I                     = sp.eye(n_features)
        L                     = self._build_graph()

        if self.gamma_i == 0.0:
            M                 = X_labeled.T @ X_labeled \
                + self.gamma_a * n_labeled_samples * I

        else:
            M                 = X_labeled.T @ X_labeled \
                + self.gamma_a * n_labeled_samples * I \
                + self.gamma_i * n_labeled_samples / n_samples**2 * self.X_.T @ L**self.p @ self.X_

        print(
            'n_samples         = {0}\n' \
            'n_labeled_samples = {1}\n' \
            'n_features        = {2}\n' \
            'n_classes         = {3}\n'.format(n_samples, n_labeled_samples, n_features, n_classes)
        )

        # Train a classifer
        self.coef_            = LA.solve(M, X_labeled.T @ y_labeled)

        return self


class RidgeClassifier(LinearLapRLSC):
    """Classifier using ridge regression.

    Parameters
    ----------
    gamma_a : float
        Regularization parameter.

    Attributes
    ----------
    X_ : array-like, shape = [n_samples, n_features], dtype = float64
        Training data.

    y_ : array-like, shape = [n_samples], dtype = float64
        Target values.

    A_ : array-like, shape = [n_samples, n_samples], dtype = float64
        Adjacency matrix.

    coef_ : array-like, shape = [n_features], dtype = float64
        Weight vector.
    """

    def __init__(self, gamma_a=1.0):
        super(RidgeClassifier, self).__init__(gamma_a=gamma_a, gamma_i=0.0)
