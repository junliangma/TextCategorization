import numpy             as np
import scipy.sparse      as sp
import matplotlib.pyplot as plt
import networkx          as nx
import seaborn           as sns
from matplotlib.colors        import ListedColormap
from sklearn.base             import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors        import kneighbors_graph, radius_neighbors_graph
from sklearn.preprocessing    import LabelBinarizer


def _fit_binary(estimator, X, y, L):
    """Fit a single binary estimator."""

    estimator                 = clone(estimator)

    return estimator.fit(X, y, L)


def _predict_binary(estimator, X, Z):
    """Make predictions using a single binary estimator."""

    return estimator.predict(X, Z)


class SemiSupervisedOneVsRestClassifier(BaseEstimator, ClassifierMixin):
    """One-vs-the-rest (OvR) multiclass strategy"""

    def __init__(
        self,         estimator,        sparsify = 'kNN', n_neighbors = 10,
        radius = 1.0, reweight = 'rbf', t        = None,  normed      = True
    ):

        self.estimator        = estimator
        self.sparsify         = sparsify
        self.n_neighbors      = n_neighbors
        self.radius           = radius
        self.reweight         = reweight
        self.t                = t
        self.normed           = normed

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Prameters
        ---------
        X : array-like, shpae = [n_samples, n_features]
            Training data.

        y : array-like, shpae = [n_samples]
            Target values (unlabeled points are marked as -1).
        """

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)

        labeled               = y != -1
        X_labeled             = X[labeled]
        y_labeled             = y[labeled]
        Y_labeled             = self.label_binarizer_.fit_transform(y_labeled)

        self.classes_         = np.unique(y_labeled)

        n_samples, n_features = X.shape
        n_labeled_samples     = y_labeled.size
        n_classes             = self.classes_.size

        X_unlabeled           = X[-labeled]
        y_unlabeled           = y[-labeled]

        if n_classes == 2:
            Y_unlabeled       = np.zeros((n_samples - n_labeled_samples, 1))
        else:
            Y_unlabeled       = np.zeros((n_samples - n_labeled_samples, n_classes))

        self.X_               = np.vstack((X_labeled, X_unlabeled))
        self.y_               = np.hstack((y_labeled, y_unlabeled))
        Y                     = np.vstack((Y_labeled, Y_unlabeled))

        L                     = self._build_graph()

        print(
            'n_samples         = {0}\n' \
            'n_labeled_samples = {1}\n' \
            'n_features        = {2}\n' \
            'n_classes         = {3}'.format(n_samples, n_labeled_samples, n_features, n_classes)
        )

        # Train classifiers
        self.estimators_      = [
            _fit_binary(self.estimator, self.X_, col, L) for col in Y.T
        ]

        return self

    def predict(self, X):
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

        Y                     = np.array([
            _predict_binary(e, X, self.X_) for e in self.estimators_
        ]).T

        return self.label_binarizer_.inverse_transform(Y)

    def plot2d(self, title=None, domain=[-1, 1], codomain=[-1, 1], predict=True):
        f, ax                 = plt.subplots()

        x1                    = np.linspace(*domain, 100)
        x2                    = np.linspace(*codomain, 100)

        n_samples, n_features = self.X_.shape
        G                     = nx.from_scipy_sparse_matrix(self.A_)
        pos                   = {i: self.X_[i] for i in range(n_samples)}
        cm_sc                 = ListedColormap(['#AAAAAA', '#FF0000', '#0000FF'])

        if title is not None:
            ax.set_title(title)

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_xlim(domain)
        ax.set_ylim(codomain)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=25, node_color=self.y_, cmap=cm_sc)

        if predict:
            xx1, xx2          = np.meshgrid(x1, x2)
            xfull             = np.c_[xx1.ravel(), xx2.ravel()]
            z                 = self.predict(xfull).reshape(100, 100)

            levels            = np.array([-1, 0, 1])
            cm_cs             = plt.cm.RdYlBu

            if self.params['gamma_i'] != 0.0:
                nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#AAAAAA')

            ax.contourf(xx1, xx2, z, levels, cmap=cm_cs, alpha=0.25)

        return (f, ax)

    def _build_graph(self):
        """Compute the graph Laplacian."""

        # Graph sparsification
        if self.sparsify == 'epsilonNN':
            self.A_           = radius_neighbors_graph(self.X_, self.radius, include_self=False)
        else:
            Q                 = kneighbors_graph(
                self.X_,
                self.n_neighbors,
                include_self  = False
            ).astype(np.bool)

            if self.sparsify   == 'kNN':
                self.A_       = (Q + Q.T).astype(np.float64)
            elif self.sparsify == 'MkNN':
                self.A_       = (Q.multiply(Q.T)).astype(np.float64)

        # Edge re-weighting
        if self.reweight == 'rbf':
            W                 = rbf_kernel(self.X_, gamma=self.t)
            self.A_           = self.A_.multiply(W)

        return sp.csgraph.laplacian(self.A_, normed=self.normed)
