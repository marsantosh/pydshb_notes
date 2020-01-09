# -*- coding: utf-8 -*-
# chapter05_linear_regression.py

# page 390 (408 of 548)

%matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#%%
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
plt.scatter(x, y)

#%%
# we can use scikit-learn's LinearRegression estimator
# to fit this data and construct the
# best fit line
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept = True)

model.fit(x[:, np.newaxis], y)

xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])

plt.figure()
plt.scatter(x, y)
plt.plot(xfit, yfit)

#%%
# the slope and intercept of the data are contained in the model's fit
# parameters, whhich in scikit-learn are always maeked by a trailing underscore
print('Model slope:\t', model.coef_[0])
print('Model intercept:\t', model.intercept_)

#%%
# multidimensional linear regression
rng = np.random.RandomState(1)
X = 10 * rng.rand(100, 3)
y = 0.5 + np.dot(X, [1.5, -2.0, 1.0])

model.fit(X, y)
print(model.intercept_)
print(model.coef_)

#%%
# Basis Function Regression
# One trick you can use to adapt linear regression to nonlinear relationships
# between variables is to transform the data according to basis functions.

# Polynomial basis functions
from sklearn.preprocessing import PolynomialFeatures
x = np.array([2, 3, 4])
poly = PolynomialFeatures(3, include_bias = False)
poly.fit_transform(x[:, None])

# we see here that the transofrmer has converted our one-dimensional array into
# a three-dimensional array by taking the exponent of each value. This new
# higher dimensional data representation can then be plugged into a 
# linnear regression

#%%
# the cleanest way to accomplish this is to use a pipeline
from sklearn.pipeline import make_pipeline
poly_model = make_pipeline(PolynomialFeatures(7),
                           LinearRegression())

rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)

poly_model.fit(x[:, np.newaxis], y)
yfit = poly_model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)

# Our linear model, through the use of 7th-order polynomial basis functions, can pro‐
# vide an excellent fit to this nonlinear data!


#%%
# Gaussian Basis Functions
# Of course, other basis functions are possible. For example, one useful pattern
# is to fit a model that is not a sum of polynomial bases, but a sum of Gassian
# bases.
from sklearn.base import BaseEstimator, TransformerMixin


class GaussianFeatures(BaseEstimator, TransformerMixin):
    '''Uniformly spaced Gaussian features for one-dimensional input.
    '''
    def __init__(self, N, width_factor = 2.0):
        self.N = N
        self.width_factor = width_factor
    
    @staticmethod
    def _gauss_basis(x, y, width, axis = None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))
    
    def fit(self, X, y = None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
    
    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
                                 self.width_, axis = 1)

gauss_model = make_pipeline(GaussianFeatures(30),
                            LinearRegression())
gauss_model.fit(x[:, np.newaxis], y)
yfit = gauss_model.predict(xfit[:, np.newaxis])

plt.figure()
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.xlim(0, 10)


#%%
# -----------------------------------------------------------------------------
#                        REGULARIZATION
# -----------------------------------------------------------------------------
# introduction of basis functions into out linear regression makes the model
# much more flexible, but it also can very quickly lead to overfitting
model = make_pipeline(GaussianFeatures(30),
                      LinearRegression())
model.fit(x[:, np.newaxis], y)

plt.figure()
plt.scatter(x, y)
plt.plot(xfit, model.predict(xfit[:, np.newaxis]))
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5)

#%%
def basis_plot(model, title = None):
    fig, ax = plt.subplots(2, sharex = True)
    model.fit(x[:, np.newaxis], y)
    ax[0].scatter(x, y)
    ax[0].plot(xfit, model.predict(xfit[:, np.newaxis]))
    ax[0].set(xlabel = 'x', ylabel = 'y', ylim = (-1.5, 1.5))
    
    if title:
        ax[0].set_title(title)
    
    ax[1].plot(model.steps[0][1].centers_,
               model.steps[1][1].coef_)
    ax[1].set(xlabel = 'basis location',
              ylabel = 'coefficient',
              xlim = (0, 10))

model = make_pipeline(GaussianFeatures(30), LinearRegression())
basis_plot(model)

# the lower panel of the figure shows the amplitud of the basis function at 
# each location. This is typical overfitting behaviour when basis functions
# overlap: the coefficients of adjacent basis functions blow up and cancel
# each other out. We know that such behaviour is problematic, and it would 
# be nice if we coudl limit such spiked explicitly in the model by penalizing
# large values of the model parameters. Such a penalty is known as
# regularization, and comes in several forms

#%%
# -----------------------------------------------------------------------------
#                        RIDGE REGRESSION
# -----------------------------------------------------------------------------
# Ridge regression (L_2 regularization) (also called Tikhonov regularization)
# This proceeds by penalizing the sum of squares (2-norms) of the model
# coefficients

from sklearn.linear_model import Ridge
model = make_pipeline(GaussianFeatures(30), Ridge(alpha = 0.1))
basis_plot(model, title = 'Ridge Regression')

# The alpha parameter is essentially a knob controlling the complexity of the
# resulting model.  In the limit alpha -> 0, we recover the standard linear
# regression result: in the limit alphaa -> infty, all model responses will
# be suppressed. One advantage of ridge repression in particular is that it
# can be computed very efficiently - at hardly more computational
# cost than the original linear regression model.

#%%
# -----------------------------------------------------------------------------
#                        LASSO REGRESSION
# -----------------------------------------------------------------------------
# This involves penalizing the sum of absolute values (1-norms) of regression
# coeffiients
from sklearn.linear_model import Lasso
model = make_pipeline(GaussianFeatures(30), Lasso(alpha = 0.001))
basis_plot(model, title = 'Lasso Regression')

#%%
#
# -----------------------------------------------------------------------------
#                        PREDICTING BICYCLE TRAFFIC
# -----------------------------------------------------------------------------
import pandas as pd
counts = pd.read_csv('data/FremontHourly.csv', index_col = 'Date', parse_dates = True)
weather = pd.read_csv('data/BicycleWeather.csv', index_col = 'DATE',
                       parse_dates = True)
#%%
# compute the total daily bicylce traffic, and put this in its own DataFrame
daily = counts.resample('d').sum()
daily['Total'] = daily.sum(axis = 1)
daily = daily[['Total']]     # remove other columns

#%%
# we sar previously that the patterns of use generally vary from day to day
# let's account for this in our data by adding binary columns that
# indicate the day ogf the week
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in range(0, 7):
    daily[days[i]] = (daily.index.dayofweek == i).astype(float)

#%%
# similarly, we might expect riders to behave differently on holidays
# let's add an indicator of this as well
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays('2012', '2016')
daily = daily.join(pd.Series(1, index = holidays, name = 'holiday'))
daily['holiday'].fillna(0, inplace = True)


#%%
# we also might suspect that the hours of daylight would affect how manu people
# ride; let's use the standard astronomical calculation to add this
# information
def hours_of_daylight(date, axis = 23.44, latitude = 47.61):
    '''Compute the hours of daylight for the diven date
    '''
    days = (date - pd.datetime(2000, 12, 21)).days
    m = (1.0 - np.tan(np.radians(latitude))
         * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
    
    return 24.0 * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.0


daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index))
daily[['daylight_hrs']].plot();

#%%
# we can also add the average temperature and total precipitation to the
# data. In addition to the inches of precipitation, let's add a flag
# that indicates whether a day is fry (has zero precipitation)

# temperatures are in 1/10 def C; convert to C
weather['TMIN'] /= 10
weather['TMAX'] /= 10
weather['Temp_C'] = 0.5 * (weather['TMIN'] + weather['TMAX'])

# precip is in 1/10 mm, convert to inches
weather['PRCP'] /= 254
weather['dry day'] = (weather['PRCP'] == 0).astype(int)

daily = daily.join(weather[['PRCP', 'Temp_C', 'dry day']])

#%%
# Finally let's add a counter that increases from day 1, and measures
# how many years have passed. This will let us measure any bserved annual
# increase pr decrease in daily crossings
daily['annual'] = (daily.index - daily.index[0]).days / 365.0

#%%
# our data is in order, let's look at it
daily.head()

#%%
# Withn  this in place, we can choose the columns to use, and fit a linear
# regression model to out data. We will set fit_intercept = False, because
# the daily flagas essentially operate as their own day-specific intercepts
column_names = [
        'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'holiday',
        'daylight_hrs', 'PRCP', 'dry day', 'Temp_C', 'annual'
]

X = daily[column_names]
y = daily['Total']

model = LinearRegression(fit_intercept = False)
model.fit(X, y)
daily['predicted'] = model.predict(X)

#%%
# finally we can compare the total and predicted bicycle traffic
# visually
daily[['Total', 'predicted']].plot(alpha = 0.5)

#%%
# It is evident that we have missed some key features, especially during
# the summer time. Either our features are not complete or there are some
# nonlinear relationships that we have failed to take into acocount.
# Nevertheless, our rought approximation is enough to give us some
# insights, and we can take a look at the coefficients of the linear model to
# estimate how  much each feature contributes to the daily dbicycle count
params = pd.Series(model.coef_, index = X.columns)
params

#%%
# These numbers are difficult to interpret without some measure og their
# uncertainty.
# we can compute these uncertainties quickly using bootstrap resamplings of
# the data
from sklearn.utils import resample
np.random.seed(1)
err = np.std([model.fit(*resample(X, y)).coef_
              for i in range(1000)], 0)

#%%
# with errors estimated, let's again look at the results
print(pd.DataFrame({'effect': params.round(0),
                    'error': err.round(0)}))

#%%
# Our model os almost certainly missing some relevant information
# For edaxample, non-linear effects (such as effects of precipitation and
# cold temperature) and nonlinear trends whithin each variable
# (such as disinclination to ride at very cold and very hot temperatures)
# cannot be accounted for in this model.
# Additionally, we have thrown away some of the finer-grained information
# (such as th difference between a rainy morning and a rany afternoon),
# and we ahve ignored correlations between days (such as the possible effect of a rainy Tuesday on Wednesday’s numbers, or the effect
# of an unexpected sunny day after a streak of rainy days).

    
# end
