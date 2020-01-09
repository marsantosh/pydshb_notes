# -*- coding: utf-8 -*-
# chapter05_naive_bayes_classification.py

'''
Naive Bayes models are a group of extremely fast and simple classification algorithms
that are often suitable for very high-dimensional datasets. Because they are so fast
and have so few tunable parameters, they end up being very useful as a quick-and-
dirty baseline for a classification problem.


In Bayesian classification, we're interested in finding the proability of a label
given some observed features, which we can write as 
P(L | features). 

If we are trying to decide between two labels, let's call them L_1 and L_2 - 
then one way to make this os tp compute the ratio of the posterior
probabilities for each label.

All we need now is some model by which we can compute P(features | L_i) 
for each label. Such a model is called a generative model because
it specifies the hypothetical random process that generates the data.
The general version of such a training step is a very difficult task, but we can
make it simpler through the use of some simplifying assumptions about
thhe form of this model.

This is were the `naive` in `naive bayes` comes in: if we make very naive
assumptions about the generative model for each lable, we can find a 
rough approximation of the generative model for each calss, and then proceed
with the Bayesian classification.

Different types of naive bayes classifiers rest on different naive assumptions
about the data, and we will examine a fre of these in the following sections.
'''

#%%
%matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#%%
# -----------------------------------------------------------------------------
#                        GAUSSIAN NAIVE BAYES
# -----------------------------------------------------------------------------
# In this classifier, the assumption is thatdata from each label is drawn
# from a simple Gaussian (normal) distribution.
from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers = 2, random_state = 2, cluster_std = 1.5)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'RdBu')

#%%
'''
One extremely fast way to create a simple model is to assume that the data
is described by a Gaussian distribution with no covariance between dimensions.
We can fit this model by simply finding the mean and standard deviation of the points
within each lavel, which is all you need to define such a distribution.

With this generative model in place for each class, we have a simple recipe
to compute the likelihood P(features | L_1) for any data point, and this we can
quickly compute the posterior ratio and determine which
label is the most probable for a given point
'''

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)

# generate some nwe data and predict the label
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)

#%%
# Now we can plot this new data to get an idea of where the devision boundary is
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'RdBu')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c = ynew, s = 20, cmap = 'RdBu', alpha = 0.1)
plt.axis(lim)

'''
We see a slightly curved boundary in the calassifications - in general, the
boundary in Gaussian naive bayes is quadratic.
'''

#%%
yprob = model.predict_proba(Xnew)
yprob[-8:].round(2)
'''
Of course, the final classification will only be as good as the model assumptions that
lead to it, which is why Gaussian naive Bayes often does not produce very good
results. Still, in many cases—especially as the number of features becomes large—this
assumption is not detrimental enough to prevent Gaussian naive Bayes from being a
useful method.
'''

#%%
# -----------------------------------------------------------------------------
#                        MULTINOMIAL NAIVE BAYES
# -----------------------------------------------------------------------------
'''
In Multinomial Naive Bayes, the features are assumed to be generated
from a simple multinomial ditribution. The multinomial distribution describes
the probability of observing counts among a number of categories, and thus
multinomial naibr bayes is most appropriate for features that represent counts
or count rates.

The idea is orecisely the same as before, except that instead of modeling the
data distribution with the best-fit Gaussian, we model the data distribution
with a best-fit multinomial distribution.
'''

#%%
'''
One place where multinomial naive bayes is often used is in text classification,
where the features are related to word counts or frequencies whithin  the documents
to be classified.
'''
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
data.target_names

# for simplicity, we will selec tjust a few of these categories and
# download the training and testing set
categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space',
              'comp.graphics']
train = fetch_20newsgroups(subset = 'train', categories = categories)
test = fetch_20newsgroups(subset = 'test', categories = categories)

print(train.data[5])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

#%%
model.fit(train.data, train.target)
labels = model.predict(test.data)

#%%
# now that we have predicted tha labels for the test data, we can evaluate
# them to learn about the performance of the estimator.
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
plt.figure()
sns.heatmap(mat.T, square = True, annot = True, fmt = 'd',
            cbar = False, xticklabels = train.target_names, yticklabels = train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')

'''
Evidently, even this very simple classifier can successfully separate space talk from
computer talk, but it gets confused between talk about religion and talk about Chris‐
tianity. This is perhaps an expected area of confusion!
'''

#%%
'''
The very cool thing here is that we now have the tools to determine the
catrgoty for any string, using the predict() method of this pipeline
'''
def predict_category(s, train = train, mdoel = model):
    pred = model.predict([s])
    return train.target_names[pred[0]]

#%%
predict_category('sendnig a payload to the ISS')

#%%
predict_category('discussing islam vs. atheism')

#%%
predict_category('determining the screen resolution')

#%%
predict_category('on the electrodynamics of bodies in movement')

#%%
predict_category('T Tauri may be a not very peculiar star')

#%%
'''
Temembrer that this is nothing more sophisticated than a simple probability
model for the (weighted) frequency of each word in the string; nevertheless
the result is striking. Even a very naive algorithm when used carefully and
trained on a large set of high-dimensional data, can be suprisingly effective.
'''


#%%
# -----------------------------------------------------------------------------
#                        WHEN TO USE NAIVE BAYES
# -----------------------------------------------------------------------------
'''
Because naive Bayesian classifiers make such stringent assumptions about data, they
will generally not perform as well as a more complicated model. That said, they have
several advantages:
    • They are extremely fast for both training and prediction
    • They provide straightforward probabilistic prediction
    • They are often very easily interpretable
    • They have very few (if any) tunable parameters
    
These advantages mean a naive Bayesian classifier is often a good choice as an initial
baseline classification. If it performs suitably, then congratulations: you have a very
fast, very interpretable classifier for your problem. If it does not perform well, then
you can begin exploring more sophisticated models, with some baseline knowledge of
how well they should perform.

Naive Bayes classifiers tend to perform especially well in one of the following
situations:
    
    • When the naive assumptions actually match the data (very rare in practice)
    • For very well-separated categories, when model complexity is less important
    • For very high-dimensional data, when model complexity is less important
    
The last two points seem distinct, but they actually are related: as the dimension of a
dataset grows, it is much less likely for any two points to be found close together
(after all, they must be close in every single dimension to be close overall). This means
that clusters in high dimensions tend to be more separated, on average, than clusters
in low dimensions, assuming the new dimensions actually add information. For this
reason, simplistic classifiers like naive Bayes tend to work as well or better than more
complicated classifiers as the dimensionality grows: once you have enough data, even
a simple model can be very powerful.
'''