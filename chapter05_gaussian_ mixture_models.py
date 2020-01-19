# -*- coding: utf-8 -*-
# chapter05_gaussian_mixture_models.py
'''
In this section we will take a look at Gaussian mixture
models, which can be viewed as an extension of the ideas behind k-means, but can
also be a powerful tool for estimation beyond simple clustering.


'''

#%%
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

#%%
# generate some data
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples = 400, centers = 4,
                       cluster_std = 0.60, random_state = 0)
X = X[:, ::-1]  # flip axes for better plotting

#%%
# plot the data with k-means labels
from sklearn.cluster import KMeans
kmeans = KMeans(4, random_state = 0)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c = labels, s = 40, cmap = 'viridis')

'''
K-means model has no instrinsic measure of probability pr uncertainty of cluster
assignments (although it may be possible to use a bootstrap approach to estimate
the uncertainty).
'''


#%%
'''
One way to think about the k-means model is that it places a circle (or, in higher
dimensions, a hyper-sphere) at the center of each cluster, with a radius defined by the
most distant point in the cluster. This radius acts as a hard cutoff for cluster assign‐
ment within the training set: any point outside this circle is not considered a member
of the cluster.
'''
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def plot_kmeans(kmeans, n_clusters = 4, rseed = 0, ax = None):
    labels = kmeans.fit_predict(X)
    
    # plot tje input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c = labels, s = 40, cmap = 'viridis', zorder = 2)
    
    # plot tje reesentation of the k-means model
    centers = kmeans.cluster_centers_
    radii = [
        cdist(X[labels == i], [center]).max()
        for i, center in enumerate(centers)
        ]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc = '#CCCCCC', lw = 3, alpha = 0.5, zorder = 1))


#%%
kmeans = KMeans(n_clusters = 4, random_state = 0)
plot_kmeans(kmeans, X)


#%%
'''
An important observation for k-means is that these cluster models must be circular: k-
means has no built-in way of accounting for oblong or elliptical clusters. So, for
example, if we take the same data and transform it, the cluster assignments end up
becoming muddled
'''
rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))

kmeans = KMeans(n_clusters = 4, random_state = 0)
plot_kmeans(kmeans, X_stretched)


#%%
# -----------------------------------------------------------------------------
#                    Generalizing E-M: Gaussian Mixture Models
# -----------------------------------------------------------------------------
'''
A Gaussian mixture model (GMM) attempts to find a mixture of multidimensional
Gaussian probability distributions that best model any input dataset. In the simplest
case, GMMs can be used for finding clusters in the same manner as k-means
'''
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components = 4).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c = labels, s = 40, cmap = 'viridis')


#%%
# But because GMM contains a probabilistic model under the hood, it is also possible
# to find probabilistic cluster assignments
probs = gmm.predict_proba(X)
print(probs[:5].round(3))

#%%
# we can visualize this uncertainty by making the size pf each point proportional
# to the uncertainty of its predicion
size = 50 * probs.max(1) ** 2
plt.scatter(X[:, 0], X[:, 1], c = labels, cmap = 'viridis', s = size)

#%%
'''
Under the hood, a Gaussian mixture model is very similar to k-means: it uses
an expectiation-maximization approach that qualitatively does the
following:
    a. E-step: for each point, find weights encoding the probability
        of membership in each cluster
    b. M-step: for each clustr, update its location, normalization, and shape
        based on all data points, making use of the weights
        
The result of this is that each cluster is associated not with a hard-edged sphere, but
with a smooth Gaussian model. Just as in the k-means expectation–maximization
approach, this algorithm can sometimes miss the globally optimal solution, and thus
in practice multiple random initializations are used.
'''

from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax = None, **kwargs):
    """Draw an ellipse with a given position and covariance."""
    ax = ax or plt.gca()
    
    # convert covatiance to princiapl axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # draw the ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label = True, ax = None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c = labels, s = 40, cmap = 'viridis',
                   zorder = 2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s = 40, zorder = 2)
    
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha = w * w_factor)
        
#%%
gmm = GaussianMixture(n_components = 4, random_state = 42)
plot_gmm(gmm, X)

#%%
'''
Similarly, we can use the GMM approach to fit out stretched dataset;
allowing for a full covariance, the model will fit even very oblong,
stretched out clusters
'''
gmm = GaussianMixture(n_components = 4, covariance_type = 'full', random_state = 42)
plot_gmm(gmm, X_stretched)


#%%
'''
Choosing the covariance type
If you look at the details of the preceding fits, you will see that the covariance_type
option was set differently within each. This hyperparameter controls the degrees of
freedom in the shape of each cluster; it is essential to set this carefully for any given
problem. The default is covariance_type="diag" , which means that the size of the
cluster along each dimension can be set independently, with the resulting ellipse
constrained to align with the axes. A slightly simpler and faster model is cova
riance_type="spherical" , which constrains the shape of the cluster such that all
dimensions are equal. The resulting clustering will have similar characteristics to that
of k-means, though it is not entirely equivalent. A more complicated and computa‐
tionally expensive model (especially as the number of dimensions grows) is to use
covariance_type="full" , which allows each cluster to be modeled as an ellipse with
arbitrary orientation.
'''

#%%
# -----------------------------------------------------------------------------
#                           GMM as Density Estimation
# -----------------------------------------------------------------------------
'''
Though GMM is often categorized as a clustering algorithm, fundamentally it is an
algorithm for density estimation. That is to say, the result of a GMM fit to some data is
technically not a clustering model, but a generative probabilistic model describing the
distribution of the data.
'''

#%%
from sklearn.datasets import make_moons
Xmoon, ymoon = make_moons(200, noise = 0.05, random_state = 0)
plt.scatter(Xmoon[:, 0], Xmoon[:, 1])

#%%
# If we try to fit this to a two-component GMM viewed as a clustering model, the
# results are not particularly useful
gmm2 = GaussianMixture(n_components = 2, covariance_type = 'full', random_state = 0)
plot_gmm(gmm2, Xmoon)


#%%
# but if we instead use many more components and ignore the cluster labels,
# we find a fit that is much closer to the input data
gmm16 = GaussianMixture(n_components = 16, covariance_type = 'full',
                        random_state = 0)
plot_gmm(gmm16, Xmoon, label = False)


#%%
'''
Here the mixture of 16 Gaussians serves not to find separated clusters of data, but
rather to model the overall distribution of the input data. This is a generative model of
the distribution, meaning that the GMM gives us the recipe to generate new random
data distributed similarly to our input. For example, here are 400 new points drawn
from this 16-component GMM fit to our original data
'''
Xnew, __ = gmm16.sample(n_samples = 400)
plt.scatter(Xnew[:, 0], Xnew[:, 1])


#%%
# -----------------------------------------------------------------------------
#                           How many components? 
# -----------------------------------------------------------------------------
'''
The fact that GMM is a generative model gives us a natural means of determining the
optimal number of components for a given dataset. A generative model is inherently
a probability distribution for the dataset, and so we can simply evaluate the likelihood
of the data under the model, using cross-validation to avoid overfitting. Another
means of correcting for overfitting is to adjust the model likelihoods using some ana‐
lytic criterion such as the Akaike information criterion (AIC) or the Bayesian infor‐
mation criterion (BIC). Scikit-Learn’s GMM estimator actually includes built-in
methods that compute both of these, and so it is very easy to operate on this
approach.
'''
n_components = np.arange(1, 21)
models = [
    GaussianMixture(n, covariance_type = 'full', random_state = 0).fit(Xmoon)
    for n in n_components
]

plt.plot(n_components, [m.bic(Xmoon) for m in models], label = 'BIC')
plt.plot(n_components, [m.aic(Xmoon) for m in models], label = 'AIC')
plt.legend(loc = 'best')
plt.xlabel('n_components')


'''
Notice the important point: this choice of number of components measures how well
GMM works as a density estimator, not how well it works as a clustering algorithm. I’d
encourage you to think of GMM primarily as a density estimator, and use it for clus‐
tering only when warranted within simple datasets.
'''

#%%
# -----------------------------------------------------------------------------
#                   Example: GMM for Generating New Data
# -----------------------------------------------------------------------------
# generate new digits data
from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape


#%%
def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize = (8, 8),
                           subplot_kw = dict(xticks = [], yticks = []))
    fig.subplots_adjust(hspace = 0.05, wspace = 0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap = 'binary')
        im.set_clim(0, 16)

plot_digits(digits.data)


#%%
'''
We have nearly 1,800 digits in 64 dimensions, and we can build a GMM on top of
these to generate more. GMMs can have difficulty converging in such a high dimen‐
sional space, so we will start with an invertible dimensionality reduction algorithm on
the data. Here we will use a straightforward PCA, asking it to preserve 99% of the
variance in the projected data
'''
from sklearn.decomposition import PCA
pca = PCA(0.99, whiten = True)
data = pca.fit_transform(digits.data)
print(data.shape)

#%%
# finnaly we can use the ivnerse transform of the PCA object to constt¡ruct the new digits
digits_new = pca.inverse_transform(data)
plot_digits(digits_new)

#%%
# -----------------------------------------------------------------------------
#                                  KDE on a Sphere
# -----------------------------------------------------------------------------
from sklearn.datasets import fetch_species_distributions

data = fetch_species_distributions()

# Get matrices/arrays of species IDs and locations
latlon = np.vstack([data.train['dd lat'],
                    data.train['dd long']]).T
species = np.array([d.decode('ascii').startswith('micro')
                    for d in data.train['species']], dtype='int')

#%%
from mpl_toolkits.basemap import Basemap
from sklearn.datasets.species_distributions import construct_grids

xgrid, ygrid = construct_grids(data)

# plot coastlnes with basemap
m = Basemap(projection = 'cyl',
            resolution = 'c',
            llcrnrlat = ygrid.min(),
            urcrnrlat = ygrid.max(),
            llcrnrlon = xgrid.min(),
            urcrnlrlon = xgrid.max())

m = drawmapboundary(fill_color = '#DDEEFF')
m.fillcontinents(color = '#FFEEDD')
m.drawcoastlines(color = 'gray', zorder = 2)
m.drawcountries(color = 'gray', zorder = 2)

# plot locations
m.scatter(latlon[:, 1], latlon[:, 0], zorder = 3,
          c = species, cmap = 'rainbow', latlon = True)


#%%
from sklearn.base import BaseEstimator, ClassifierMixin

class KDEClassifier(BaseEstimator, ClassifierMixin):
    """Bayesian generative classification based on KDE.
    
    parameters
    ----------
    bandwidth: float
        the kernel bandwidth within each class
    kernel: str
        the kernel name, passed to KernelDensity
    """
    def __init__(self, bandwidth = 1.0, kernel = 'gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
    
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth = self.bandwidth,
                                      kernel = self.kernel).fit(Xi)
                        for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        
        return self
    
    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims = True)
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


#%%
# Using our custom estimator
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV

digits = load_digits()

bandwidths = 10 ** np.linspace(0, 2, 100)
grid = GridSearchCV(KDEClassifier(), {'bandwidth': bandwidths})
grid.fit(digits.data, digits.target)

#%%
scores = {key: val for key, val in grid.cv_results_.items()}
print(scores)