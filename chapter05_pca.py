# -*- coding: utf-8 -*-
# chapter05_pca.py

#%%
# -----------------------------------------------------------------------------
#                                  PCA
# -----------------------------------------------------------------------------
#
'''
PCA is fundamentally a
dimensionality reduction algorithm, but it can also be useful as a tool for visualiza‐
tion, for noise filtering, for feature extraction and engineering, and much more. After
a brief conceptual discussion of the PCA algorithm, we will see a couple examples of
these further applications.
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#%%
# consider the following 200 points
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal')

'''
Rather than attempting to predict the y values from the x values, the
unsupervised learning problem attempts to learn about the relationship between
the x and y values.

In PCA, one quantifies the relationship by finding a list og the principal
axes in the data, and using those axes to describe the dataset.
'''
#%%
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(X)
print(pca)

'''
The fit learns some quantities from the data, most importantly the components
and explained variance
'''
print('pca.components_: ', pca.components_)
print('pca.explained_variance_: ', pca.explained_variance_)

#%%
'''
To see what these numbers mean, let's visualize them as vectors over the input
data, using the 'components' to define the direction of the vectorm, and the
'explained variance' to define the squared-length of the vector
'''
def draw_vector(v0, v1, ax = None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle = '->',
                      linewidth = 2,
                      shrinkA = 0,
                      shrinkB = 0,
                      color = 'k')
    ax.annotate('', v1, v0, arrowprops = arrowprops)

#%%
# plot data
plt.scatter(X[:, 0], X[:, 1], alpha = 0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal')

'''
These vectors represent the principal axes of the data, and the length shown
is an indication of how `important` that axis is in describing the
distribution of the data - more precisely, it is a measure of the variance
of the data when projected onto that axis.
The projection of each data point onto the principal axes are the 
`principal components` of the data.

This transformation from data axes to principal axes is as an affine transformation,
which basically means it is composed of a translation, rotation and
uniform scaling.


'''

#%%
# -----------------------------------------------------------------------------
#                        PCA as dimensionality reduction
# -----------------------------------------------------------------------------
#
'''
Using PCA for dimensionality reduction involves zeroing out one or more of the
smallest principal components, resulting in a lower-dimensional projection of the
data that preserves the maximal data variance.
'''
pca = PCA(n_components = 1)
pca.fit(X)
X_pca = pca.transform(X)
print('original shape: ', X.shape)
print('transformed shape: ', X_pca.shape)

#%%
X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha = 0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha = 0.8)
plt.axis('equal')

'''
PCA dimensionality reduction means removing information along the least important
principal axis or axes,  leaving only the component(s) of the data with the highest
variance. The fraction of variance that is cut out is roughly a 
measure of how much 'information' is discarded in this reduction of 
dimensionality.
This reduced-dimension dataset is in some senses “good enough” to encode the most
important relationships between the points.
'''
#%%
# -----------------------------------------------------------------------------
#                     PCA for visualization: Handwritten digits
# -----------------------------------------------------------------------------
#
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)

#%%
pca = PCA(n_components = 2)   # project from 64 to 2 dimensions
projected = pca.fit_transform(digits.data)
print(digits.data.shape)
print(projected.shape)

#%%
# we can now plot tje first two principal components of each point to learn
# about the data
plt.scatter(projected[:, 0], projected[:, 1],
            c = digits.target, edgecolor = 'none',
            alpha = 0.5, cmap = plt.cm.get_cmap('Spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()

'''
PCA can be thought of as a process of choosing optimal basis functions, such
that adding together just the first few of them is enough to suitably
reconstruct the bulk of the elements in the dataset. The principal components, which
act as the low-dimensional representation of our data, are simply the
coefficients that multiply each of the elements in this series.
'''

#%%
# -----------------------------------------------------------------------------
#                        Choosing the number of components
# -----------------------------------------------------------------------------
#
# cummulative explained variance ratio as a function of the number
# of components
pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


#%%
# -----------------------------------------------------------------------------
#                               PCA as Noise Filtering
# -----------------------------------------------------------------------------
#
'''
PCA can also be used as a filtering approach for noisy data. The idea is this:
components with variance much larger than the effect of the noise should
be relatively unaffected by the noise. 
'''
# plot several of the inpit noise-free data
def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize = (10, 4),
                             subplot_kw = {'xticks':[], 'yticks': []},
                             gridspec_kw = dict(hspace = 0.1, wspace = 0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap = 'binary', interpolation = 'nearest',
                  clim = (0, 16))

plot_digits(digits.data)

#%%
# add random noise to create a noisy dataset and replot it
np.random.seed(42)
noisy = np.random.normal(digits.data, 4)
plot_digits(noisy)

#%%
# fit pca on noisy data
pca = PCA(0.50).fit(noisy)
print(pca.n_components_)

# here 50% of the variance amounts to 12 principal coponents
# now we compute these components, and then use the inverse of the transform to
# reconstruct the filtered digits
components = pca.transform(noisy)
filtered = pca.inverse_transform(components)
plot_digits(filtered)

'''
This signal preserving/noise filtering property makes PCA a very useful feature selec‐
tion routine—for example, rather than training a classifier on very high-dimensional
data, you might instead train the classifier on the lower-dimensional representation,
which will automatically serve to filter out random noise in the inputs
'''

#%%
# -----------------------------------------------------------------------------
#                                Example: Eigenfaces
# -----------------------------------------------------------------------------
#
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person = 60)
print(faces.target_names)
print(faces.images.shape)

#%%
# take a look at the principal axes that span this dataset
# RandomizedPCA contains a randomized method to approximate the first N
# principal components much more quickly thatn the standard PCA estimator,
# and thus is very useful for high-dmensional data
from sklearn.decomposition import PCA
pca = PCA(n_components = 150, svd_solver = 'randomized')
pca.fit(faces.data)

#%%
# visualize images with their pricipal components
fig, axes = plt.subplots(3, 8, figsize = (9, 4),
                         subplot_kw = {'xticks': [], 'yticks': []},
                         gridspec_kw = dict(hspace = 0.1, wspace = 0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(62, 47), cmap = 'bone')
    
#%%
# cumulative variance of the components
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

#%%
# compute components and projected faces
pca = PCA(n_components = 50, svd_solver ='randomized').fit(faces.data)
components = pca.transform(faces.data)
projected = pca.inverse_transform(components)

#%%
# plot the results
fig, ax = plt.subplots(2, 10, figsize = (10, 2.5),
                       subplot_kw = {'xticks': [], 'yticks': []},
                       gridspec_kw = dict(hspace = 0.1, wspace = 0.1))

for i in range(0, 10):
    ax[0, i].imshow(faces.data[i].reshape(62, 47), cmap = 'binary_r')
    ax[1, i].imshow(projected[i].reshape(62, 47), cmap = 'binary_r')

ax[0, 0].set_ylabel('full-dim\ninput')
ax[1, 0].set_ylabel('150-dim\nreconstruction')

'''
although it reduces the dimensionality of
the data by nearly a factor of 20, the projected images contain enough information
that we might, by eye, recognize the individuals in the image. What this means is that
our classification algorithm needs to be trained on 150-dimensional data rather than
3,000-dimensional data, which depending on the particular algorithm we choose, can
lead to a much more efficient classification.
'''

#%%
# -----------------------------------------------------------------------------
#                                Summary
# -----------------------------------------------------------------------------
#
'''
In this section we have discussed the use of principal component analysis for dimen‐
sionality reduction, for visualization of high-dimensional data, for noise filtering, and
for feature selection within high-dimensional data. Because of the versatility and
interpretability of PCA, it has been shown to be effective in a wide variety of contexts
and disciplines. Given any high-dimensional dataset, I tend to start with PCA in
order to visualize the relationship between points (as we did with the digits), to
understand the main variance in the data (as we did with the eigenfaces), and to
understand the intrinsic dimensionality (by plotting the explained variance ratio).
Certainly PCA is not useful for every high-dimensional dataset, but it offers a
straightforward and efficient path to gaining insight into high-dimensional data.
PCA’s main weakness is that it tends to be highly affected by outliers in the data. For
this reason, many robust variants of PCA have been developed, many of which act to
iteratively discard data points that are poorly described by the initial components.
Scikit-Learn contains a couple interesting variants on PCA, including RandomizedPCA
and SparsePCA , both also in the sklearn.decomposition submodule. Randomi
zedPCA , which we saw earlier, uses a nondeterministic method to quickly approxi‐
mate the first few principal components in very high-dimensional data, while
SparsePCA introduces a regularization term that serves to enforce
sparsity of the components.
'''