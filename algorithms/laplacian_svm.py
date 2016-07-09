import numpy        as np
import scipy.sparse as sp
import scipy.linalg as LA
from cvxopt                   import matrix, solvers
from sklearn.metrics.pairwise import pairwise_kernels

from .base                    import BaseMR, MRClassifierMixin


class LapSVC(BaseMR, MRClassifierMixin):
    """Laplacian Support Vector Machines Classifier.

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
        Y                     = sp.diags(y_labeled)
        J                     = sp.eye(n_labeled_samples, n_samples)
        K                     = pairwise_kernels(self.X_, metric=self.kernel, **self.kwds)
        L                     = self._build_graph()

        if self.gamma_i == 0.0:
            M                 = (2 * self.gamma_a * I).toarray()

        else:
            M                 = 2 * self.gamma_a * I \
                + 2 * self.gamma_i / n_samples**2 * L**self.p @ K

        print(
            'n_samples         = {0}\n' \
            'n_labeled_samples = {1}\n' \
            'n_features        = {2}\n' \
            'n_classes         = {3}\n'.format(n_samples, n_labeled_samples, n_features, n_classes)
        )

        # Construct the QP, invoke solver
        sol                   = solvers.qp(
            P                 = matrix(Y @ J @ K @ LA.inv(M) @ J.T @ Y),
            q                 = matrix(-1 * np.ones(n_labeled_samples)),
            G                 = matrix(np.vstack((
                -1 * np.eye(n_labeled_samples),
                n_labeled_samples * np.eye(n_labeled_samples)
            ))),
            h                 = matrix(np.hstack((
                np.zeros(n_labeled_samples),
                np.ones(n_labeled_samples)
            ))),
            A                 = matrix(y_labeled, (1, n_labeled_samples), 'd'),
            b                 = matrix(0.0)
        )

        # Train a classifer
        self.dual_coef_       = LA.solve(M, J.T @ Y @ np.array(sol['x']).ravel())

        return self


class SVC(LapSVC):
    """C-Support Vector Classification.

    Parameters
    ----------
    gamma_a : float
        Regularization parameter.

    kernel : {'rbf', 'poly', 'linear'}
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
        super(SVC, self).__init__(
            gamma_a = gamma_a, gamma_i = 0.0, kernel = kernel, gamma_k = gamma_k,
            degree  = degree,  coef0   = coef0
        )
