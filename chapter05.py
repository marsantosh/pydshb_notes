#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# chapter05.py

#%%
import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()

#%%
%matplotlib
sns.set()
sns.pairplot(iris, hue = 'species', height = 1.5)

#%%
X_iris = iris.drop('species', axis = 1)
X_iris.shape

#%%
y_iris = iris['species']
y_iris.shape

#%%
# Suppervised learning example: Simple linear regression
import matplotlib.pyplot as plt
import numpy as np

plt.figure()
rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y)

# choose a class of model
from sklearn.linear_model import LinearRegression

# choose model hyperpatameters
model = LinearRegression(fit_intercept = True)
model

# arrange dat ainto features amtrix and target vector
X = x[:, np.newaxis]
X.shape

# fit model to the data
model.fit(X, y)
print(model.coef_)
print(model.intercept_)

# predict labels for unkown data
xfit = np.linspace(-1, 11)

Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

# visualize results
plt.scatter(x, y)
plt.plot(xfit, yfit)

#%%
# Supervised learning example: iris calssification
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state = 1)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)

#%% Unsupervised learning example_ Iris dimensionality
from sklearn.decomposition import PCA
model = PCA(n_components = 2)   # isntantiate model with hyperparameters
model.fit(X_iris)               # fit to data
X_2D = model.transform(X_iris)  # transform the data to two dimensions

# plot the results
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot('PCA1', 'PCA2', hue = 'species', data = iris, fit_reg = False)

# we see that in the two-dimensional representation the species are
# fairly well separated, even though the PCA algorithm had no knowledge
# of the species labels


#%%
# unsupervised learning: Iris clustering (Gaussian Mixture Model)
from sklearn.mixture import GaussianMixture as GMM
model = GMM(n_components = 3,
            covariance_type = 'full')
model.fit(X_iris)
y_gmm = model.predict(X_iris)

iris['cluster'] = y_gmm
sns.lmplot('PCA1', 'PCA2', data = iris, hue = 'species',
           col = 'cluster', fit_reg = False)


#%%
# Application: Exploring Handwritten Digits
from sklearn.datasets import load_digits
digits = load_digits()
digits.images.shape

# lets visualize the first hundred
import matplotlib.pyplot as plt
fig, axes = plt.subplots(
    10, 10, figsize = (10, 10),
    subplot_kw = {'xticks': [], 'yticks': []},
    gridspec_kw = dict(hspace = 0.1, wspace = 0.1)
)

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap = 'binary', interpolation = 'nearest')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform = ax.transAxes, color = 'green')
    
X = digits.data
print('X.shape:', X.shape)

y = digits.target
print('y.shape:', y.shape)

# dimensionality reduction
# use Isomap to reduce dimension 64 -> 2
from sklearn.manifold import Isomap
iso = Isomap(n_components = 2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
data_projected.shape

plt.figure()
plt.scatter(data_projected[:, 0], data_projected[:, 1],
            c = digits.target, edgecolors='none', alpha=0.7,
            cmap = plt.cm.get_cmap('Paired')
            )
plt.colorbar(label = 'digit label', ticks = range(10))
plt.clim(-0.5, 9.5)

#%%
# Classification on digits
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state = 0)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

# gauge accuracy bu comparing true values of the test set to the 
# predictions
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, y_model)

sns.heatmap(mat, square = True, annot = True, cbar = False)
plt.xlabel('predicted value')
plt.ylabel('true value')

#%%
# plot the inputs with their respected labels
fig, axes = plt.subplots(10, 10, figsize = (8, 8),
                         subplot_kw = {'xticks': [], 'yticks': []},
                         gridspec_kw = dict(hspace = 0.1, wspace = 0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap = 'binary', interpolation = 'nearest')
    ax.text(0.05, 0.05, str(y_model[i]),
            transform = ax.transAxes,
            color = 'green' if (ytest[i] == y_model[i]) else 'red')
    
#%%
# hyperparameters and model validation
# Model Validation the wrong way
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)

model.fit(X, y)
y_model = model.predict(X)

from sklearn.metrics import accuracy_score
print(accuracy_score(y, y_model))

#%% Model validation the rightway: Holdout sets
from sklearn.model_selection import train_test_split

# split the data with .50 in each set
X1, X2, y1, y2 = train_test_split(X, y, random_state = 0, train_size = 0.5)

# fit the model on one set of data
model.fit(X1, y1)

# evaluate the model on the second set of data
y2_model = model.predict(X2)
accuracy_score(y2, y2_model)

#%%
# Using cross validation
y2_model = model.fit(X1, y1).predict(X2)
y1_model = model.fit(X2, y2).predict(X1)
print(accuracy_score(y1, y1_model))
print(accuracy_score(y2, y2_model))

# using sklearns implemented cross-vañ
from sklearn.model_selection import cross_val_score
print(cross_val_score(model, X, y, cv = 5))

#%%
# LeaveOneOut cross-valudation
from sklearn.model_selection import LeaveOneOut
scores = cross_val_score(model, X, y, cv = LeaveOneOut())
print(scores)
'''
Because we have 150 samples, the leave-one-out cross-validation yields scores for 150
trials, and the score indicates either successful (1.0) or unsuccessful (0.0) prediction.
Taking the mean of these gives an estimate of the error rate:
'''
print(scores.mean())

#%%
# Validation curves in sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def PolynomialRegression(degree = 2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

import numpy as np

def make_data(N, err = 1.0, rseed = 1):
    # rnadomly sample the data
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1.0 / (X.ravel() + 0.1)
    
    if err > 0:
        y += err * rng.randn(N)
    
    return X, y

X, y = make_data(40)

#%%
# we can now visualize our data, along with polynomial fits of several degrees
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

X_test = np.linspace(-0.1, 1.1, 500)[:, None]

plt.scatter(X.ravel(), y, color = 'black')
axis = plt.axis()

for degree in [1, 2, 3, 5]:
    y_test = PolynomialRegression(degree).fit(X, y).predict(X_test)
    plt.plot(X_test.ravel(), y_test, label = f'degree={degree}')

plt.xlim(-1, 1.5)
plt.ylim(-2, 12)
plt.legend(loc = 'best')

#%%
# Validation curve
from sklearn.model_selection import validation_curve
degree = np.arange(0, 21)

train_score, val_score = validation_curve(PolynomialRegression(), X, y,
                                          'polynomialfeatures__degree',
                                          degree, cv = 7)

plt.plot(degree, np.median(train_score, 1), color = 'blue', label = 'training score')
plt.plot(degree, np.median(val_score, 1), color = 'red', label = 'validation score')
plt.legend(loc = 'best')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')

#%%
# from the valdiation curve, we can read off that the optimal trade-off between
# bias and variance is found for a third-order polynomial
plt.figure()
plt.scatter(X.ravel(), y)
lim = plt.axis()
y_test = PolynomialRegression(3).fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_test)
plt.axis(lim)


#%%
# Learning Curves
X2, y2 = make_data(200)
plt.figure()
plt.scatter(X2.ravel(), y2)

plt.figure()
degree = np.arange(21)
train_score2, val_score2 = validation_curve(PolynomialRegression(), X2, y2,
                                           'polynomialfeatures__degree',
                                           degree, cv = 7)
plt.plot(degree, np.median(train_score2, 1), color = 'blue',
         label = 'training score')
plt.plot(degree, np.median(val_score2, 1), color = 'red', label = 'validation score')
plt.plot(degree, np.median(train_score, 1), color = 'blue', alpha = 0.3,
         linestyle = 'dashed')
plt.plot(degree, np.median(val_score, 1), color = 'red', alpha = 0.3,
         linestyle = 'dashed')
plt.legend(loc = 'lower center')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')

# A plot of the training/validation score with respect to the size of the 
# training set is knows as a learning curve

'''
The general behavior we would expect from a learning curve is this:
    
    • A model of a given complexity will overit a small dataset: this means the training
        score will be relatively high, while the validation score will be relatively low.
    • A model of a given complexity will underit a large dataset: this means that the
        training score will decrease, but the validation score will increase.
    • A model will never, except by chance, give a better score to the validation set than
        the training set: this means the curves should keep getting closer together but
        never cross.
'''

#%%
# Learning curves in Scikit-learn
from sklearn.model_selection import learning_curve

fig, ax = plt.subplots(1, 2, figsize = (16, 6))
fig.subplots_adjust(left = 0.0625, right = 0.95, wspace = 0.1)

for i, degree in enumerate([2, 9]):
    N, train_lc, val_lc = learning_curve(PolynomialRegression(degree),
                                         X, y, cv = 7,
                                         train_sizes=np.linspace(0.3, 1, 25))
    ax[i].plot(N, np.mean(train_lc, 1), color = 'blue', label = 'training score')
    ax[i].plot(N, np.mean(val_lc, 1), color = 'red', label = 'validation score')
    ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], [-1], color = 'gray',
      linestyle = 'dashed')
    ax[i].set_ylim(0, 1)
    ax[i].set_xlim(N[0], N[-1])
    ax[i].set_xlabel('training size')
    ax[i].set_ylabel('score')
    ax[i].set_title(f'degree = {degree}', size = 14)
    ax[i].legend(loc = 'best')
    
    
#%%
# Validation in practice: Grid search
from sklearn.model_selection import GridSearchCV

param_grid = {
    'polynomialfeatures__degree': np.arange(21),
    'linearregression__fit_intercept': [True, False],
    'linearregression__normalize': [True, False]
}
grid = GridSearchCV(PolynomialRegression(), param_grid, cv = 7)

# calling the fit method
grid.fit(X, y)

print(grid.best_params_)

# finally if we wish, we can use the best model and show the fit to our
# data using code from before
model = grid.best_estimator_

plt.scatter(X.ravel(), y)
lim = plt.axis()
y_test = model.fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_test)
plt.axis(lim)


#%%
# Feature Engineering
# categorical features
data = [
    {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
    {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
    {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
    {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse = False, dtype = int)
vec.fit_transform(data)

#%%
# inspect feature names
vec.get_feature_names()
#%%
# There is one clear disadvantage of this approach: if your categogry has
# many possible values, this can greatly increase the size of your dataset
# however, because the encoded data contains mostyl zeros, a sparse output
# can be a very efficient solution
vec = DictVectorizer(sparse = True, dtype = int)
vec.fit_transform(data)

#%%
# Text features
'''
Another common need in feature engineering is to convert text to a set of representa‐
tive numerical values. For example, most automatic mining of social media data relies
on some form of encoding the text as numbers. One of the simplest methods of
encoding data is by word counts: you take each snippet of text, count the occurrences
of each word within it, and put the results in a table.
'''
sample = [
        'problem of evil',
        'evil queen',
        'horizon problem'
        ]

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
X = vec.fit_transform(sample)
print(X)

#%%
import pandas as pd
pd.DataFrame(X.toarray(), columns = vec.get_feature_names())

#%%
'''
There are some issues with this approach, however, the raw word counts lead
to featuees that put too much weight on words that appear very
frequently, and this can be suboptimal in some classification algorithms,
One approach to fix this is knows as `term frequency-inverse document frequency (TF-IDF),
which weights the word counts by a measure of how often they appear in the
documents.
'''
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns = vec.get_feature_names())

#%%
# derived features
%matplotlib
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])
plt.scatter(x, y)

from sklearn.linear_model import LinearRegression
X = x[:, np.newaxis]
model = LinearRegression().fit(X, y)
yfit = model.predict(X)
plt.scatter(x, y)
plt.plot(x, yfit)

#%%
'''
It is clear that we need a more sophisticated model to describe the
relationship between x and y. We can do this by transofrming the data, adding
extra columns of features to drive more flexibility in the model
'''
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree =3, include_bias = False)
X2 = poly.fit_transform(X)
print(X2)

model = LinearRegression().fit(X2, y)
yfit = model.predict(X2)
plt.scatter(x, y)
plt.plot(x, yfit)


#%%
# Imputation of missing data
import numpy as np
X = np.array(
    [
     [np.nan, 0, 3],
     [3, 7, 9],
     [3, 5, 2],
     [4, np.nan, 6],
     [8, 8, 1]
    ]
)
y = np.array([14, 16, -1, 8, -5])

'''
When applying a typical machine learning model to such data, we will need to first
replace such missing data with some appropriate fill value. This is known as imputa‐
tion of missing values, and strategies range from simple (e.g., replacing missing values
with the mean of the column) to sophisticated (e.g., using matrix completion or a
robust model to handle such data).

For a baseline imputation approach, using the mean, median, or most
frequent value, Scikit-Learn provides the Imputer class:
'''
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy = 'mean')
X2 = imp.fit_transform(X)
print(X2)

# this imputed data can then be fed directly into a model
model = LinearRegression().fit(X2, y)
model.predict(X2)

#%%
# Using pipeline
from sklearn.pipeline import make_pipeline
model = make_pipeline(SimpleImputer(strategy = 'mean'),
                      PolynomialFeatures(degree = 2),
                      LinearRegression())

model.fit(X, y)
print(y)
print(model.predict(X))