from abc                      import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import networkx          as nx
import numpy             as np
import seaborn           as sns
import scipy.sparse      as sp
from matplotlib.colors        import ListedColormap
from sklearn.base             import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors        import kneighbors_graph, radius_neighbors_graph


class BaseMR(BaseEstimator, metaclass=ABCMeta):
    """Base class for manifold regularization."""

    def __init__(
        self,               gamma_a = 1.0, gamma_i  = 1.0,   kernel   = 'rbf',
        gamma_k     = None, degree  = 3,   coef0    = 1.0,   sparsify = 'kNN',
        n_neighbors = 1,    radius  = 1.0, reweight = 'rbf', t        = None,
        normed      = True, p       = 1
    ):
        self.gamma_a          = gamma_a
        self.gamma_i          = gamma_i
        self.kernel           = kernel
        self.degree           = degree
        self.coef0            = coef0
        self.gamma_k          = gamma_k
        self.sparsify         = sparsify
        self.n_neighbors      = n_neighbors
        self.radius           = radius
        self.reweight         = reweight
        self.t                = t
        self.normed           = normed
        self.p                = p

        if self.kernel   == 'rbf':
            self.kwds         = {'gamma': self.gamma_k}

        elif self.kernel == 'poly':
            self.kwds         = {'gamma': self.gamma_k, 'degree': self.degree, 'coef0': self.coef0}

        elif self.kernel == 'linear':
            self.kwds         = {}

    @abstractmethod
    def fit(self, X, y):
        """Fit the model according to the given training data."""

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
            W                 = pairwise_kernels(self.X_, metric='rbf', **{'gamma': self.t})
            self.A_           = self.A_.multiply(W)

        return sp.csgraph.laplacian(self.A_, normed=self.normed)


class ClassifierMixin:
    """Mixin class for all classifiers in scikit-learn."""

    def plot2d(self, title=None, domain=[-1, 1], codomain=[-1, 1], predict=True):
        f, ax                 = plt.subplots()

        x1                    = np.linspace(*domain, 100)
        x2                    = np.linspace(*codomain, 100)

        n_samples, n_features = self.X_.shape
        G                     = nx.from_scipy_sparse_matrix(self.A_)
        pos                   = {i: self.X_[i] for i in range(n_samples)}
        cm_sc                 = ListedColormap(['#FF0000', '#AAAAAA', '#0000FF'])

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
            cm_cs             = plt.cm.RdBu

            if self.gamma_i != 0.0:
                nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#AAAAAA')

            ax.contourf(xx1, xx2, z, levels, cmap=cm_cs, alpha=0.25)

        return (f, ax)


class MRClassifierMixin(ClassifierMixin):
    """Mixin for manifold regularization classifiers."""

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features], dtype = float64
            Samples.

        Returns
        -------
        y : array-like, shape = [n_samples], dtype = float64
            Predictions for input data.
        """

        K                     = pairwise_kernels(self.X_, X, metric=self.kernel, **self.kwds)

        return np.sign(self.dual_coef_ @ K)


class LinearMRClassifierMixin(ClassifierMixin):
    """Mixin for linear manifold regularization classifiers."""

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like, shape = [n_samples, n_features], dtype = float64
            Samples.

        Returns
        -------
        y : array-like, shape = [n_samples], dtype = float64
            Predictions for input data.
        """

        return np.sign(X @ self.coef_)
