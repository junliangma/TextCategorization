import numpy        as np
import scipy.sparse as sp
import scipy.linalg as LA
from sklearn.metrics.pairwise import pairwise_kernels

from .base                    import BaseMR, MRClassifierMixin


class LapRLSC(BaseMR, MRClassifierMixin):
    """Laplacian Regularized Least Squares Classifier.

    Parameters
    ----------
    gamma_a : float
        Regularization parameter.

    gamma_i : float
        Smoothness regularization parameter.

    kernel : {'rbf', 'poly', 'linear'}
        Kernel type.

    gamma_k : float
        Kernel coefficient for 'rbf' and 'poly'.

    degree : integer > 0
        Degree of the polynomial kernel function.

    coef0 : float
        Independent term in kernel function.

    sparsify : {'kNN', 'MkNN', 'epsilonNN'}
        Graph sparsification type.

    n_neighbors : integer > 0
        Number of neighbors for each sample.

    radius : float
        Radius of neighborhoods.

    reweight : {'binary', 'rbf'}
        Edge re-weighting type.

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

    dual_coef_ : array-like, shape = [n_samples], dtype = float64
        Weight vector in kernel space.

    References
    ----------
    Mikhail Belkin, Partha Niyogi, Vikas Sindhwani,
    "On Manifold Regularization",
    AISTATS, 2005.
    """

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
        I                     = sp.eye(n_samples)
        J                     = sp.diags(labeled, dtype=bool)
        K                     = pairwise_kernels(self.X_, metric=self.kernel, **self.kwds)
        L                     = self._build_graph()

        if self.gamma_i == 0.0:
            M                 = J @ K \
                + self.gamma_a * n_labeled_samples * I

        else:
            M                 = J @ K \
                + self.gamma_a * n_labeled_samples * I \
                + self.gamma_i * n_labeled_samples / n_samples**2 * L**self.p @ K

        print(
            'n_samples         = {0}\n' \
            'n_labeled_samples = {1}\n' \
            'n_features        = {2}\n' \
            'n_classes         = {3}\n'.format(n_samples, n_labeled_samples, n_features, n_classes)
        )

        # Train a classifer
        self.dual_coef_       = LA.solve(M, self.y_)

        return self


class KernelRidgeClassifier(LapRLSC):
    """Classifier using kernel ridge regression.

    Parameters
    ----------
    gamma_a : float
        Regularization parameter.

    kernel : {'rbf', 'poly', 'lienar'}
        Kernel type.

    gamma_k : float
        Kernel coefficient for 'rbf' and 'poly'.

    degree : integer > 0
        Degree of the polynomial kernel function.

    coef0 : float
        Independent term in kernel function.

    Attributes
    ----------
    X_ : array-like, shape = [n_samples, n_features], dtype = float64
        Training data.

    y_ : array-like, shape = [n_samples], dtype = float64
        Target values.

    A_ : array-like, shape = [n_samples, n_samples], dtype = float64
        Adjacency matrix.

    dual_coef_ : array-like, shape = [n_samples], dtype = float64
        Weight vector in kernel space.
    """

    def __init__(self, gamma_a=1.0, kernel='rbf', gamma_k=None, degree=3, coef0=1.0):
        super(KernelRidgeClassifier, self).__init__(
            gamma_a = gamma_a, gamma_i = 0.0, kernel = kernel, gamma_k = gamma_k,
            degree  = degree,  coef0   = coef0
        )
