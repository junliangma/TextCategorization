import numpy        as np
import scipy.sparse as sp
import scipy.linalg as LA
from cvxopt                   import matrix, solvers
from sklearn.base             import BaseEstimator
from sklearn.metrics.pairwise import rbf_kernel

from .base                    import MRBinaryClassifierMixin
from .multiclass              import SemiSupervisedOneVsRestClassifier


class BinaryLapSVC(BaseEstimator, MRBinaryClassifierMixin):
    """Laplacian Support Vectior Machines."""

    def fit(self, X, y, L):
        """Fit the model according to the given training data.

        Prameters
        ---------
        X : array-like, shpae = [n_samples, n_features]
            Training data.

        y : array-like, shpae = [n_samples]
            Target values (unlabeled points are marked as 0).

        L : array-like, shpae = [n_samples, n_samples]
            Graph Laplacian.
        """

        labeled               = y != 0
        y_labeled             = y[labeled]
        n_samples, n_features = X.shape
        n_labeled_samples     = y_labeled.size
        I                     = sp.eye(n_samples)
        Y                     = sp.diags(y_labeled)
        J                     = sp.eye(n_labeled_samples, n_samples)
        K                     = rbf_kernel(X, gamma=self.gamma_k)
        M                     = 2 * self.gamma_a * I \
            + 2 * self.gamma_i / n_samples**2 * L**self.p @ K

        # Construct the QP, invoke solver
        solvers.options['show_progress'] = False
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


class LapSVC(SemiSupervisedOneVsRestClassifier):
    """Laplacian Support Vectior Machines.

    Parameters
    ----------
    gamma_a : float
        Regularization parameter.

    gamma_i : float
        Smoothness regularization parameter.

    gamma_k : float
        Kernel coefficient.

    sparsify : {'kNN', 'MkNN', 'epsilonNN'}
        Graph sparsification type.

    n_neighbors : int > 0
        Number of neighbors for each sample.

    radius : float
        Radius of neighborhoods.

    reweight: {'rbf', 'binary'}
        Edge re-weighting type.

    t : float
        Kernel coefficient.

    normed : boolean, dealut True
        If True, then compute normalized Laplacian.

    p : integer > 0
        Degree of the graph Laplacian.

    Attributes
    ----------
    X_ : array-like, shape = [n_samples, n_features]
        Training data.

    y_ : array-like, shape = [n_samples]
        Target values.

    classes_ : array-like, shpae = [n_classes]
        Class labels.

    A_ : array-like, shape = [n_samples, n_samples]
        Adjacency matrix.

    estimators_ : list of n_classes estimators
        Estimators used for predictions.

    label_binarizer_ : LabelBinarizer object
        Object used to transform multiclass labels to binary labels and vice-versa.

    References
    ----------
    Mikhail Belkin, Partha Niyogi, Vikas Sindhwani,
    "On Manifold Regularization",
    AISTATS, 2005.
    """

    def __init__(
        self,             gamma_a     = 1.0,  gamma_i = 1.0, gamma_k  = 1.0,
        sparsify = 'kNN', n_neighbors = 10,   radius  = 1.0, reweight = 'rbf',
        t        = None,  normed      = True, p       = 1
    ):

        super(LapSVC, self).__init__(
            estimator   = BinaryLapSVC(), sparsify = sparsify,
            n_neighbors = n_neighbors,    radius   = radius,
            reweight    = reweight,       t        = t,
            normed      = normed
        )

        self.params           = {
            'gamma_a': gamma_a, 'gamma_i': gamma_i, 'gamma_k': gamma_k, 'p': p
        }

        self.estimator.set_params(**self.params)
