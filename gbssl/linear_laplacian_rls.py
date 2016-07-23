import numpy        as np
import scipy.sparse as sp
import scipy.linalg as LA
from sklearn.base import BaseEstimator

from .base        import LMRBinaryClassifierMixin
from .multiclass  import SemiSupervisedOneVsRestClassifier


class BinaryLinearLapRLSC(BaseEstimator, LMRBinaryClassifierMixin):
    """Linear Laplacian Regularized Least Squares Classifier."""

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
        X_labeled             = X[labeled]
        y_labeled             = y[labeled]
        n_samples, n_features = X.shape
        n_labeled_samples     = y_labeled.size
        I                     = sp.eye(n_features)
        M                     = X_labeled.T @ X_labeled \
            + self.gamma_a * n_labeled_samples * I \
            + self.gamma_i * n_labeled_samples / n_samples**2 * X.T @ L**self.p @ X

        # Train a classifer
        self.coef_            = LA.solve(M, X_labeled.T @ y_labeled)

        return self


class LinearLapRLSC(SemiSupervisedOneVsRestClassifier):
    """Linear Laplacian Regularized Least Squares Classifier.

    Parameters
    ----------
    gamma_a : float
        Regularization parameter.

    gamma_i : float
        Smoothness regularization parameter.

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

    A_ : array-like, shape = [n_samples, n_samples]
        Adjacency matrix.

    classes_ : array-like, shpae = [n_classes]
        Class labels.

    estimators_ : list of n_classes estimators
        Estimators used for predictions.

    label_binarizer_ : LabelBinarizer object
        Object used to transform multiclass labels to binary labels and vice-versa.

    References
    ----------
    Vikas Sindhwani, Partha Niyogi, Mikhail Belkin,
    "Linear Manifold Regularization for Large Scale Semi-supervised Learning",
    Proc. of 22nd ICML Workshop on Learning with Partially Classified Training data, 2005.
    """

    def __init__(
        self,               gamma_a = 1.0, gamma_i  = 1.0,   sparsify = 'kNN',
        n_neighbors = 10,   radius  = 1.0, reweight = 'rbf', t        = None,
        normed      = True, p       = 1
    ):

        super(LinearLapRLSC, self).__init__(
            estimator   = BinaryLinearLapRLSC(), sparsify = sparsify,
            n_neighbors = n_neighbors,           radius   = radius,
            reweight    = reweight,              t        = t,
            normed      = normed
        )

        self.params           = {'gamma_a': gamma_a, 'gamma_i': gamma_i, 'p': p}

        self.estimator.set_params(**self.params)
