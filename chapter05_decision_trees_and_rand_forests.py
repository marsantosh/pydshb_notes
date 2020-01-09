# -*- coding: utf-8 -*-
# chapter05_decision_trees_and_rand_forests.py
'''
Random Forests are an example of an ensamble method,
a methot that relies on aggregating the results of an ensemble
of simple estimators. The somewhat surprising result iwith such ensemble
methods is that the sym can be greater than the parts; that is, a 
majority vote among a nubmer of estimators can end up being better than any
of the individual estimators doing the voting. <cool_stuff>

'''

#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#%%
# -----------------------------------------------------------------------------
#                    Motivating Random Forests: Decision Trees
# -----------------------------------------------------------------------------
#
'''
In machine learning implementations of
decision trees, the questions generally take the form of axis-aligned splits in the data;
that is, each node in the tree splits the data into two groups using a cutoff value
within one of the features. Let’s now take a look at an example.
'''

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples = 300, centers = 4,
                  random_state = 0, cluster_std = 1.0)
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'rainbow')

'''
A simple decision tree built on this data will iteratively split the data along
one or the other axis according to some quantitative criterion, and at each
level assign the label of the new region according to a majority vote
of points whithin it.
'''

#%%
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier().fit(X, y)

#%%
def visualize_classifier(model, X, y, ax = None, cmap = 'rainbow'):
    ax = ax or plt.gca()
    
    # plot thhe training points
    ax.scatter(X[:, 0], X[:, 1], c = y, s = 30, cmap = cmap,
               clim = (y.min(), y.max()), zorder = 3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num = 200),
                         np.linspace(*ylim, num = 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha = 0.3,
                           levels = np.arange(n_classes + 1) - 0.5,
                           cmap = cmap, clim = (y.min(), y.max()),
                           zorder = 1)
    
    ax.set(xlim = xlim, ylim = ylim)


#%%
visualize_classifier(DecisionTreeClassifier(), X, y)

#%%
'''
Overfitting turns out to be a general property of decision trees;
it is very easy to go too deep in the tree, and thus to
fit details of the particular data rather than the
overall properties of the distributions they are drawn from.


'''

#%%
# -----------------------------------------------------------------------------
#                    Ensembles of Estimators: Random Forests
# -----------------------------------------------------------------------------
#
'''
This notion—that multiple overfitting estimators can be combined to reduce the
effect of this overfitting—is what underlies an ensemble method called bagging. Bag‐
ging makes use of an ensemble (a grab bag, perhaps) of parallel estimators, each of
which overfits the data, and averages the results to find a better classification. An
ensemble of randomized decision trees is known as a random forest.
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

tree = DecisionTreeClassifier()
bag = BaggingClassifier(tree, n_estimators = 100, max_samples = 0.8,
                        random_state = 1)
bag.fit(X, y)
visualize_classifier(bag, X, y)

'''
In this example, we have randomized the data by fitting each estimator with a ran‐
dom subset of 80% of the training points. In practice, decision trees are more effec‐
tively randomized when some stochasticity is injected in how the splits are chosen;
this way, all the data contributes to the fit each time, but the results of the fit still have
the desired randomness.
'''

#%%
# sklearn implementation of RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100, random_state = 0)
visualize_classifier(model, X, y)

'''
We see that by averaging over 100 randomly perturbed models, we end up with an
overall model that is much closer to our intuition about how the parameter space
should be split.
'''

#%%
# -----------------------------------------------------------------------------
#                             Random Foresrt Regression
# -----------------------------------------------------------------------------
#
# Random forests can also be made to work in the case of regression
#
# consider the following data, drawn from the combination of a fast and
# slow osicllation
rng = np.random.RandomState(42)
x = 10 * rng.rand(200)

def model(x, sigma = 0.3):
    fast_oscillation = np.sin(5 * x)
    slow_oscillation = np.sin(0.5 * x)
    noise = sigma * rng.randn(len(x))
    
    return slow_oscillation + fast_oscillation + noise

y = model(x)
plt.errorbar(x, y, 0.3, fmt = 'o')

#%%
# using the random forest regressor, we can find the best-fit curve
# as follows
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(200)
forest.fit(x[:, None], y)

xfit = np.linspace(0, 10, 1000)
yfit = forest.predict(xfit[:, None])
ytrue = model(xfit, sigma = 0)

plt.errorbar(x, y, 0.3, fmt = 'o', alpha = 0.5)
plt.plot(xfit, yfit, '-r')
plt.plot(xfit, ytrue, '-k', alpha = 0.5)

'''
As you can see, the nonparametric random forest model
is flexible enough to fit the multiperiod data, without us
needing to specify a multiperiod model!
'''

#%%
# -----------------------------------------------------------------------------
#                  Example: Random Forest for Classifying Digits
# -----------------------------------------------------------------------------
#
from sklearn.datasets import load_digits
digits = load_digits()
digits.keys()

#%%
# visualize some points
fig = plt.figure(figsize=(6, 6)) # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the digits: each image is 8x8 pixels
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))
    
#%%
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target,
                                                random_state = 0)
model = RandomForestClassifier(n_estimators = 1000)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

#%%
# classification report
from sklearn.metrics import classification_report
print(classification_report(ytest, ypred))

#%%
# confussion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest, ypred))


#%%
# -----------------------------------------------------------------------------
#                          Summary of Random Forests
# -----------------------------------------------------------------------------
#
'''
Random forests are a powerful method with several advantages:

+ Both training and prediction are very fast, because of the simplicity of the under‐
lying decision trees. In addition, both tasks can be straightforwardly parallelized,
because the individual trees are entirely independent entities.

+ The multiple trees allow for a probabilistic classification: a majority vote among
estimators gives an estimate of the probability (accessed in Scikit-Learn with the
predict_proba() method).

+ The nonparametric model is extremely flexible, and can thus perform well on
tasks that are underfit by other estimators.


A primary disadvantage of random forests is that the results are not easily interpreta‐
ble; that is, if you would like to draw conclusions about the meaning of the classifica‐
tion model, random forests may not be the best choice.
'''