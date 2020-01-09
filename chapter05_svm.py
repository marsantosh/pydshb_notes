# -*- coding: utf-8 -*-
# chapter05_svm.py

'''
A generative classification si a model describing the distribution
of each underlying class to probabilistically determine
labels for new points.

A discriminative classification simply finds a line or curve or manifold
that divides the classes from each other.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# use seaborn plotting defaults
import seaborn as sns; sns.set()
#%%
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples = 50, centers = 2,
                  random_state = 0, cluster_std = .60)
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'autumn')

#%%
# A linear discriminative classifier would attempt to draw a straight line
# separating the two sets of data, and thereby create a model for 
# classification.

xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'autumn')
plt.plot([0.6], [2.1], 'x', color = 'red', markeredgewidth = 2,
         markersize = 10)

for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, '-k')
    
plt.xlim(-1, 3.5)

#%%
#%%
# -----------------------------------------------------------------------------
#                  Support Vector Machines: Maximizing the Margin
# -----------------------------------------------------------------------------
'''
Support vector machines offer one way to improve this (drawing border curves
between classes)
The intution is this: rather than simply drawing a zero-width line between
the classes, we can draw around each line a margin of some width, up to the
nearest point. Here is an example of how this might look
'''
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = 'autumn')

for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor = 'none',
                     color = '#AAAAAA', alpha = 0.4)
plt.xlim(-1, 3.5)

# in support vector machines, the line that maximizes this margin is the one
# we wil chose as the optimal model. Support vector machines are an example
# of such maximum margin estimator

#%%
# Fitting a support vector machine
from sklearn.svm import SVC
model = SVC(kernel = 'linear', C = 1E10)
model.fit(X, y)

#%%
# to better visualize what's happening here, let's create a quick convenience 
# function that will plot SVM decision boudnaries for us
def plot_svc_decision_function(model, ax = None, plot_support = True):
    '''Plot the decision function for a two-dimensional SVC
    '''
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create a grid to evaluate the mdoel
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors = 'k',
               levels = [-1, 0, 1], alpha = 0.5,
               linestyles = ['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s = 300, linewidth = 1,
                   facecolors = 'none')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'autumn')
plot_svc_decision_function(model)

#%%
model.support_vectors_
'''
A key to this classifier’s success is that for the fit, only the position of the support vec‐
tors matters; any points further from the margin that are on the correct side do not
modify the fit! Technically, this is because these points do not contribute to the loss
function used to fit the model, so their position and number do not matter so long as
they do not cross the margin.
'''

def plot_svm(N = 10, ax = None):
    X, y = make_blobs(n_samples = 200, centers =2,
                      random_state = 0, cluster_std = 0.60)
    X = X[:N]
    y = y[:N]
    model = SVC(kernel = 'linear', C = 1E10)
    model.fit(X, y)
    
    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'autumn')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 6)
    plot_svc_decision_function(model, ax)
    
fig, ax = plt.subplots(1, 2, figsize = (16, 6))
fig.subplots_adjust(left = 0.0625, right = 0.95, wspace = 0.1)
for axi, N in zip(ax, [60, 120]):
    plot_svm(N, axi)
    axi.set_title(f'N = {N}')

#%%

from ipywidgets import interact, fixed
interact(plot_svm, N = [10, 200], ax = fixed(None))
#%%
'''
In the left panel we see the model and the support vectors for 60 training
points. In the right panel, we have doubled the number of training points but
the model has not changed: the htree support vectors from the left panel
are still the support vectors from the right panel.
This insensivity to the exact behaviour of distant points is one of the
strenghts of the SVM model.
'''


#%%
# -----------------------------------------------------------------------------
#                  Beyond linear boundaries: Kernel SVM
# -----------------------------------------------------------------------------
'''
Where SVM becomes extremely powerful is when it is combined with kernels.
'''
from sklearn.datasets import make_circles
X, y = make_circles(100, factor = 0.1, noise = 0.1)

clf = SVC(kernel = 'linear').fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'autumn')
plot_svc_decision_function(clf, plot_support = False)

#%%
'''
Its clear that no lienar discrimination will ever be able to separate this
data.
We can think about how we might project the dat ainto a higher diension
such that a linear separator would be sufficient.
For example, one simple projection we could sue would be to compute
a radias basis function centered on the middle clump
'''
r = np.exp(-(X ** 2).sum(1))

#%%
from ipywidgets import interact, fixed
from mpl_toolkits import mplot3d

def plot_3d(elev = 30, azim = 30, X = X, y = y):
    ax = plt.subplot(projection = '3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c = y, s = 50, cmap = 'autumn')
    ax.view_init(elev = elev, azim = azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')


interact(plot_3d, elev = [-90, 90], azip = (-180, 180),
         X = fixed(X), y = fixed(y))

#%%
'''
We can say that with this additional dimension, the data becomes
trivially linearly separable bu drawing a separating plane at,
say r = 0.7

Here we had to choose and carefully tune our projection; if we had not centered our
radial basis function in the right location, we would not have seen such clean, linearly
separable results. In general, the need to make such a choice is a problem: we would
like to somehow automatically find the best basis functions to use.

One strategy to this end is to compute a basis function centered at every point in the
dataset, and let the SVM algorithm sift through the results. This type of basis function
transformation is known as a kernel transformation, as it is based on a similarity rela‐
tionship (or kernel) between each pair of points.

A potential problem with this strategy—projecting N points into N dimensions—is
that it might become very computationally intensive as N grows large. However,
because of a neat little procedure known as the kernel trick, a fit on kernel-
transformed data can be done implicitly—that is, without ever building the full N-
dimensional representation of the kernel projection! This kernel trick is built into the
SVM, and is one of the reasons the method is so powerful.
'''

#%%
clf = SVC(kernel = 'rbf', C = 1E6)
clf.fit(X, y)

#%%
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s = 300, lw = 1, facecolors = 'none')

'''
Using this kernelized support vector machine, we learn a suitable nonlinear decision
boundary. This kernel transformation strategy is used often in machine learning to
turn fast linear methods into fast nonlinear methods, especially for models in which
the kernel trick can be used.
'''

#%%
# Tuning the SVM: Softening margins
#
# overlap data
X, y = make_blobs(n_samples = 100, centers = 2, random_state = 0,
                  cluster_std = 1.2)
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'autumn')

'''
To handle this case, the SVM implementation has a bit of a fudge-factor that “softens”
the margin; that is, it allows some of the points to creep into the margin if that allows
a better fit. The hardness of the margin is controlled by a tuning parameter, most
often known as C. For very large C, the margin is hard, and points cannot lie in it. For
smaller C, the margin is softer, and can grow to encompass some points.
'''

#%%
X, y = make_blobs(n_samples = 100, centers = 2,
                  random_state = 0, cluster_std = 0.8)

fig, ax = plt.subplots(1, 2, figsize = (16, 6))
fig.subplots_adjust(left = 0.0625, right = 0.95, wspace = 0.1)

for axi, C in zip(ax, [10.0, 0.1]):
    model = SVC(kernel = 'linear', C = C).fit(X, y)
    axi.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'autumn')
    plot_svc_decision_function(model, axi)
    axi.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s = 300, lw = 1, facecolors = 'none')
    axi.set_title('C = {0:.1f}'.format(C), size = 14)
plt.show()

'''
The optimal value of the C parameter will depend on your dataset, and should be
tuned via cross-validation or a similar procedure.
'''


#%%
# -----------------------------------------------------------------------------
#                         Face recognition example
# -----------------------------------------------------------------------------
#
from PIL import Image
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person = 60)
print(faces.target_names)
print(faces.images.shape)

#%%
# plot a few of these faces to see wht we're working with
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap = 'bone')
    axi.set(xticks = [], yticks = [],
            xlabel = faces.target_names[faces.target[i]]
            )

#%%
# doing PCA to reduce dimensionality and getting fundamental components
# of the data
'''
RandomizePCA was depreciated in an older version of SKLearn and is simply a parameter in PCA.
'''
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

pca = PCA(n_components = 150, whiten = True, random_state = 42)
svc = SVC(kernel = 'rbf', class_weight = 'balanced')
model = make_pipeline(pca, svc)

#%%
# splitting data
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target,
                                                random_state = 42)

#%%
# explore combinations of parameters
from sklearn.model_selection import GridSearchCV
param_grid = {'svc__C': [1, 5, 10, 50],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid, verbose = 2, n_jobs = 6).fit(Xtrain, ytrain)
print(grid.best_params_)

#%%
'''
The optimal values fall toward the middle of our grid; if they fell at the edges, we
would want to expand the grid to make sure we have found the true optimum.
Now with this cross-validated model, we can predict the labels for the test data, which
the model has not yet seen
'''
model = grid.best_estimator_
yfit = model.predict(Xtest)

#%%
# take a look at a few of the test images along with their
# predicted values
fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap = 'bone')
    axi.set(xticks = [], yticks = [])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color = 'black' if yfit[i] == ytest[i] else 'red')

fig.suptitle('Predicted Names; Incorrect labels in red', size = 14)

#%%
from sklearn.metrics import classification_report
print(classification_report(ytest, yfit,
                            target_names = faces.target_names))

#%%
# confussion matrix
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square = True, annot = True, fmt = 'd',
            cbar = False, xticklabels = faces.target_names,
            yticklabels = faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')

#%%
# -----------------------------------------------------------------------------
#                             SVM pros and cons
# -----------------------------------------------------------------------------
#
'''
                            ============ PROS ============
+ Their dependence on relatively few support vetors means that they
are very compact models, and take up very little memory

+ Once the model is trained, the prediction phase is very fast

+ Because they are affected only by points near the margin, they work well
with high-dimensional data - even data with more dimensions than samples,
which is a challending regime for other algoritms

+ their integration with kernel methods makes them very versatile, able to
adapt to many types of data

                            ============ CONS ============
+ The scaling with the number of samples N is O[N^3] at worst, or O[N^2] for effi‐
cient implementations. For large numbers of training samples, this computa‐
tional cost can be prohibitive.

+ The results are strongly dependent on a suitable choice for the softening parame‐
ter C. This must be carefully chosen via cross-validation, which can be expensive
as datasets grow in size.

+ The results do not have a direct probabilistic interpretation. This can be estima‐
ted via an internal cross-validation (see the probability parameter of SVC ), but
this extra estimation is costly.
'''
