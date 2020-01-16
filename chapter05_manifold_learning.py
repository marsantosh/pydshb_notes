# -*- coding: utf-8 -*-
# chapter05_manifold_learning.py
'''
While PCA is flexible, fast, and easily
interpretable, it does not perform so well when there are nonlinear relationships
within the data

To address this deficiency, we can turn to a class of methods known as manifold learn‐
ing—a class of unsupervised estimators that seeks to describe datasets as low-
dimensional manifolds embedded in high-dimensional spaces. When you think of a
manifold, I’d suggest imagining a sheet of paper: this is a two-dimensional object that
lives in our familiar three-dimensional world, and can be bent or rolled in two
dimensions. In the parlance of manifold learning, we can think of this sheet as a two-
dimensional manifold embedded in three-dimensional space.

Rotating, reorienting, or stretching the piece of paper in three-dimensional space
doesn’t change the flat geometry of the paper: such operations are akin to linear
embeddings. If you bend, curl, or crumple the paper, it is still a two-dimensional
manifold, but the embedding into the three-dimensional space is no longer linear.
Manifold learning algorithms would seek to learn about the fundamental two-
dimensional nature of the paper, even as it is contorted to fill the three-dimensional
space.

'''
#%%
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

#%%
# -----------------------------------------------------------------------------
#                            Manifold Learning
# -----------------------------------------------------------------------------
#
def make_hello(N = 1000, rseed = 42):
    # make a plot with "HELLO" text; save as png
    fig, ax = plt.subplots(figsize = (4, 1))
    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
    ax.axis('off')
    ax.text(
        0.5, 0.4, 'HELLO', va = 'center', ha = 'center', weight = 'bold',
        size = 85
    )
    fig.savefig('hello.png')
    plt.close(fig)
    
    # open this png and draw random points from it
    from matplotlib.image import imread
    data = imread('hello.png')[::-1, :, 0].T
    rng = np.random.RandomState(rseed)
    X = rng.rand(4 * N, 2)
    i, j = (X * data.shape).astype(int).T
    mask = (data[i, j] < 1)
    X = X[mask]
    
    X[:, 0] *= (data.shape[0] / data.shape[1])
    X = X[:N]
    return X[np.argsort(X[:, 0])]


X = make_hello(1000)
colorize = dict(c = X[:, 0], cmap = plt.cm.get_cmap('rainbow', 5))
plt.scatter(X[:, 0], X[:, 1], **colorize)
plt.axis('equal')


#%%
# -----------------------------------------------------------------------------
#                            Manifold Learning
# -----------------------------------------------------------------------------
#
'''
Looking at data like this, we can see that the particular choice of x and y values of the
dataset are not the most fundamental description of the data: we can scale, shrink, or
rotate the data, and the “HELLO” will still be apparent. For example, if we use a rota‐
tion matrix to rotate the data, the x and y values change, but the data is still funda‐
mentally the same
'''

def rotate(X, angle):
    theta = np.deg2rad(angle)
    R = [
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ]
    return np.dot(X, R)


X2 = rotate(X, 20) + 5
plt.scatter(X2[:, 0], X2[:, 1], **colorize)
plt.axis('equal')

#%%
'''
This tells us that the x and y values are not necessarily fundamental to the relation‐
ships in the data. What is fundamental, in this case, is the distance between each point
and the other points in the dataset. A common way to represent this is to use a dis‐
tance matrix: for N points, we construct an N × N array such that entry i, j contains
the distance between point i and point j. Let’s use Scikit-Learn’s efficient pair
wise_distances function to do this for our original data:
'''
from sklearn.metrics import pairwise_distances
D = pairwise_distances(X)
print(D.shape)

#%%
# visualize the matrix
plt.imshow(D, zorder = 2, cmap = 'Blues', interpolation = 'nearest')
plt.colorbar()

#%%
# if we similarly construct a distance matrix for our rotated and translated
# data, we see that it is the same
D2 = pairwise_distances(X2)
print(np.allclose(D, D2))

#%%
'''
Further, while computing this distance matrix from the (x, y) coordinates is straight‐
forward, transforming the distances back into x and y coordinates is rather difficult.
This is exactly what the multidimensional scaling algorithm aims to do: given a dis‐
tance matrix between points, it recovers a D-dimensional coordinate representation
of the data. Let’s see how it works for our distance matrix, using the precomputed
dissimilarity to specify that we are passing a distance matrix
'''
from sklearn.manifold import MDS
model = MDS(n_components = 2, dissimilarity = 'precomputed', random_state = 1)
out = model.fit_transform(D)
plt.scatter(out[:, 0], out[:, 1], **colorize)
plt.axis('equal')

'''
The MDS algorithm recovers one of the possible two-dimensional coordinate repre‐
sentations of our data, using only the N × N distance matrix describing the relation‐
ship between the data points.
'''
#%%
# -----------------------------------------------------------------------------
#                            MMDS as manifold Learning
# -----------------------------------------------------------------------------
#

'''
The usefulness of this becomes more apparent when we consider the fact that dis‐
tance matrices can be computed from data in any dimension. So, for example, instead
of simply rotating the data in the two-dimensional plane, we can project it into three
dimensions using the following function
'''
def random_projection(X, dimension = 3, rseed = 42):
    assert dimension >= X.shape[1]
    rng = np.random.RandomState(rseed)
    C = rng.randn(dimension, dimension)
    e, V = np.linalg.eigh(np.dot(C, C.T))
    return np.dot(X, V[:X.shape[1]])

X3 =random_projection(X, 3)
print(X3.shape)

#%%
# visualize points
from mpl_toolkits import mplot3d
ax = plt.axes(projection = '3d')
ax.scatter3D(X3[:, 0], X3[:, 1], X3[:, 2],
             **colorize)
ax.view_init(azim = 70, elev = 50)

#%%
'''
We can now ask the MDS estimator to input this three-dimensional data, compute the
distance matrix, and then determine the optimal two-dimensional embedding for this
distance matrix. The result recovers a representation of the original data
'''
model = MDS(n_components = 2, random_state = 1)
out3 = model.fit_transform(X3)
plt.scatter(out3[:, 0], out3[:, 1], **colorize)
plt.axis('equal')

#%%
'''
This is essentially the goal of a manifold learning estimator: given high-dimensional
embedded data, it seeks a low-dimensional representation of the data that preserves
certain relationships within the data. In the case of MDS, the quantity preserved is the
distance between every pair of points.
'''


#%%
# -----------------------------------------------------------------------------
#                       Nonlinear Embeddings: Where MDS Fails
# -----------------------------------------------------------------------------
#
'''
Our discussion so far has considered linear embeddings, which essentially consist of
rotations, translations, and scalings of data into higher-dimensional spaces. Where
MDS breaks down is when the embedding is nonlinear—that is, when it goes beyond
this simple set of operations. Consider the following embedding, which takes the
input and contorts it into an “S” shape in three dimensions:
'''
def make_hello_s_curve(X):
    t = (X[:, 0] - 2) * 0.75 * np.pi
    x = np.sin(t)
    y = X[:, 1]
    z = np.sign(t) * (np.cos(t) - 1)
    return np.vstack((x, y, z)).T


XS = make_hello_s_curve(X)

#%%
# visualize it
from mpl_toolkits import mplot3d
ax = plt.axes(projection = '3d')
ax.scatter3D(XS[:, 0], XS[:, 1], XS[:, 2],
             **colorize)

#%%
'''
If we try a simple MDS algorithm on this data, it is not able to “unwrap” this nonlin‐
ear embedding, and we lose track of the fundamental relationships in the embedded
manifold.
'''
from sklearn.manifold import MDS
model = MDS(n_components = 2, random_state = 2)
plt.figure()
outS = model.fit_transform(XS)
plt.scatter(outS[:, 0], outS[:, 1], **colorize)
plt.axis('equal')

'''
The best two-dimensional linear embedding does not unwrap the S-curve, but
instead throws out the original y-axis.
'''

#%%
# -----------------------------------------------------------------------------
#                  Nonlinear Manifolds: Locally Linear Embedding
# -----------------------------------------------------------------------------
#
'''
How can we move forward here? Stepping back, we can see that the source of the
problem is that MDS tries to preserve distances between faraway points when con‐
structing the embedding. But what if we instead modified the algorithm such that it
only preserves distances between nearby points? The resulting embedding would be
closer to what we want.


locally linear embedding
(LLE): rather than preserving all distances, it instead tries to preserve only the distan‐
ces between neighboring points: in this case, the nearest 100 neighbors of each point.
'''

#%%
'''
LLE comes in a number of flavors; here we will use the modiied LLE algorithm to
recover the embedded two-dimensional manifold. In general, modified LLE does bet‐
ter than other flavors of the algorithm at recovering well-defined manifolds with very
little distortion
'''
from sklearn.manifold import LocallyLinearEmbedding
model = LocallyLinearEmbedding(n_neighbors = 100, n_components = 2,
                               method = 'modified', eigen_solver = 'dense')
out = model.fit_transform(XS)

fig, ax = plt.subplots()
ax.scatter(out[:, 0], out[:, 1], **colorize)
ax.set_ylim(0.15, -0.15)

'''
The result remains somewhat distorted compared to our original manifold, but cap‐
tures the essential relationships in the data!
'''

#%%
# -----------------------------------------------------------------------------
#                   Some Thoughts on Manifold Methods
# -----------------------------------------------------------------------------
#
'''
Though this story and motivation is compelling, in practice manifold learning tech‐
niques tend to be finicky enough that they are rarely used for anything more than
simple qualitative visualization of high-dimensional data.
The following are some of the particular challenges of manifold learning, which all
contrast poorly with PCA:
    
• In manifold learning, there is no good framework for handling missing data. In
contrast, there are straightforward iterative approaches for missing data in PCA.
• In manifold learning, the presence of noise in the data can “short-circuit” the
manifold and drastically change the embedding. In contrast, PCA naturally filters
noise from the most important components.
• The manifold embedding result is generally highly dependent on the number of
neighbors chosen, and there is generally no solid quantitative way to choose an
optimal number of neighbors. In contrast, PCA does not involve such a choice.
• In manifold learning, the globally optimal number of output dimensions is diffi‐
cult to determine. In contrast, PCA lets you find the output dimension based on
the explained variance.
• In manifold learning, the meaning of the embedded dimensions is not always
clear. In PCA, the principal components have a very clear meaning.

'''


#%%
# -----------------------------------------------------------------------------
#                           Example: Isomap on Faces
# -----------------------------------------------------------------------------
#
#%%
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person = 30)
print(faces.data.shape)

#%%
# visualize the images
fig, ax = plt.subplots(4, 8, subplot_kw = dict(xticks = [], yticks = []))
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap = 'gray')

#%%
'''
We would like to plot a low-dimensional embedding of the 2,914-dimensional data to
learn the fundamental relationships between the images. One useful way to start is to
compute a PCA, and examine the explained variance ratio, which will give us an idea
of how many linear features are required to describe the data
'''
from sklearn.decomposition import PCA
model = PCA(100, svd_solver = 'randomized').fit(faces.data)
plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.xlabel('n components')
plt.ylabel('cumulative variance')

#%%
'''
We see that for this data, nearly 100 components are required to preserve 90% of the
variance. This tells us that the data is intrinsically very high dimensional—it can’t be
described linearly with just a few components.

When this is the case, nonlinear manifold embeddings like LLE and Isomap can be
helpful. We can compute an Isomap embedding on these faces using the same pattern
shown before:
'''
from sklearn.manifold import Isomap
model = Isomap(n_components = 2)
proj = model.fit_transform(faces.data)
print(proj.shape)


#%%
'''
The output is a two-dimensional projection of all the input images. To get a better
idea of what the projection tells us, let’s define a function that will output image
thumbnails at the locations of the projections:
'''
from matplotlib import offsetbox

def plot_components(data, model, images = None, ax = None,
                    thumb_frac = 0.05, cmap = 'gray'):
    ax = ax or plt.gca()
    
    proj = model.fit_transform(data)
    ax.plot(proj[:, 0], proj[:, 1], '.k')
    
    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        
        for i in range(0, data.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist)  < min_dist_2:
                # dont show points that are too close
                continue
            
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap = cmap),
                proj[i])
            
            ax.add_artist(imagebox)

#%%
fig, ax = plt.subplots(figsize = (10, 10))
plot_components(faces.data,
                model = Isomap(n_components = 2),
                images = faces.images[:, ::2, ::2])

'''
The result is interesting: the first two Isomap dimensions seem to describe global
image features: the overall darkness or lightness of the image from left to right, and
the general orientation of the face from bottom to top. This gives us a nice visual
indication of some of the fundamental features in our data.
'''

#%%
# -----------------------------------------------------------------------------
#                     Example: Visualizing Structure in Digits
# -----------------------------------------------------------------------------
#
from sklearn.datasets import load_digits
mnist = load_digits()
print(mnist.data.shape)

#%%
fig, ax = plt.subplots(6, 8, subplot_kw = dict(xticks = [], yticks = []))
for i, axi in enumerate(ax.flat):
    axi.imshow(mnist.data[i].reshape(8, 8), cmap = 'gray_r')

#%%
data = mnist.data[::]
target = mnist.target[::]

model = Isomap(n_components = 2)
proj = model.fit_transform(data)
plt.scatter(proj[:, 0], proj[:, 1], c = target, cmap = plt.cm.get_cmap('jet', 10))
plt.colorbar(ticks = range(0, 10))

#%%
'''
The resulting scatter plot shows some of the relationships between the data points,
but is a bit crowded. We can gain more insight by looking at just a single number at a
time
'''
from sklearn.manifold import Isomap

# choose digit 1
data = mnist.data[mnist.target == 1][::]
fig, ax = plt.subplots(figsize = (10, 10))
model = Isomap(n_neighbors = 5, n_components = 2, eigen_solver = 'dense')
plot_components(data, model, images = data.reshape(-1, 8, 8),
                ax = ax, thumb_frac = 0.05, cmap = 'gray_r')


'''
The result gives you an idea of the variety of forms that the number “1” can take
within the dataset. The data lies along a broad curve in the projected space, which
appears to trace the orientation of the digit. As you move up the plot, you find ones
that have hats and/or bases, though these are very sparse within the dataset. The pro‐
jection lets us identify outliers that have data issues (i.e., pieces of the neighboring
digits that snuck into the extracted images).
'''

