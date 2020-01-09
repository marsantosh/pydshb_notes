# chapter04.py
# matplotlib

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#%%
%matplotlib
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)

plt.scatter(x, y, c = colors, s = sizes, alpha = 0.3,
            cmap = 'viridis')

plt.colorbar()
plt.show()

#%%
from sklearn.datasets import load_iris
iris = load_iris()
features = iris.data.T

plt.scatter(features[0], features[1], alpha = 0.2,
            s = 100 * features[3], c = iris.target, cmap = 'viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

#%%

# for any scientific measurement, accurate accounting for errors is nearly as
# important, if not more importnat, than accurate reporting of the number
# itself.

# basic errorbars
plt.style.use('seaborn-whitegrid')
x = np.linspace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)
plt.errorbar(x, y, yerr = dy, fmt = '.k', capsize = 2, ecolor = 'red')

#%%
# Continuous Errors
from sklearn.gaussian_process import GaussianProcessRegressor

# define the model and draw some data
model = lambda x:x * np.sin(x)
xdata = np.array([1, 3, 5, 6, 8])
ydata = model(xdata)

# compute the gaussian process fit
gp = GaussianProcessRegressor(corr)


#%%
# Density and Contour Plots
def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)


#%%
contours = plt.contour(X, Y, Z, 3, colors = 'black')
plt.clabel(contours, inline = True, fontsize = 8)

plt.imshow(Z, extent = [0, 5, 0, 5], origin = 'lower',
           cmap = 'RdGy', alpha = 0.5)
plt.colorbar()

#%%
x1 = np.random.normal(0, 0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)
colors = ['red', 'green', 'orange']

kwargs = dict(histtype = 'stepfilled', alpha = 0.3, normed = True, bins = 40)

plt.figure()
plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs)


#%%
#
# Two dimmensional histograms and binnings
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T

plt.hist2d(x, y, bins = 30, cmap = 'Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')

#%%
# plt.hexbin Hexagonal binnings
plt.hexbin(x, y, gridsize = 30, cmap = 'Blues')
cb = plt.colorbar(label = 'count in bin')

#%%
# Kernel Density Estimation
from scipy.stats import gaussian_kde

# git an array of sie [Bdim, Nsamples]
data = np.vstack([x, y])
kde = gaussian_kde(data)

# evaluate on a regular grid
xgrid = np.linspace(-3.5, 3.5, 40)
ygrid = np.linspace(-6, 6, 40)
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

# plot the result as an image
plt.imshow(Z.reshape(Xgrid.shape),
           origin = 'lower',
           aspect = 'auto',
           extent = [-3.5, 3.5, -6, 6],
           cmap = 'Blues'
           )

cb = plt.colorbar()
cb.set_label('density')

#%%
# Custopmizing Plot Legends
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
# plt.style.use('classic')
import numpy as np

x = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), '-b', label = 'Sine')
ax.plot(x, np.cos(x), '--r', label = 'Cosine')
ax.axis('equal')
leg = ax.legend()
# ax.legend(loc = 'upper left', frameon = True)
ax.legend(frameon = False, loc = 'lower center', ncol = 2)
# ax.legend(fancybox = True)


#%%
# multiple legends
fig, ax = plt.subplots()

lines = []
styles = ['-', '--', '-.', ':']
x = np.linspace(0, 10, 1000)

for i in range(0, 4):
    lines += ax.plot(x, np.sin(x - i * np.pi / 2), 
                     styles[i], color = 'black')

ax.axis('equal')

# specify the liubnes abd labels of the first legend
ax.legend(lines[:2], ['line A', 'line B'],
          loc = 'upper right', frameon = False)

# create the second legend and add the artist manually
from matplotlib.legend import Legend
leg = Legend(ax, lines[2:], ['line C', 'line D'],
             loc = 'lower right', frameon = False)

ax.add_artist(leg)


#%%
# Handwritten digits
import matplotlib.pyplot as plt
%matplotlib
from sklearn.datasets import load_digits
digits = load_digits(n_class = 6)

fig, ax = plt.subplots(8, 8, figsize = (6, 6))
for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap = 'binary')
    axi.set(xticks = [], yticks = [])

#%%
# project the figits into 2 dimensions using Isomap
plt.figure()
from sklearn.manifold import Isomap
iso = Isomap(n_components = 2)
projection = iso.fit_transform(digits.data)

# plot the results
plt.scatter(projection[:, 0], projection[:, 1], lw = 0.1,
            c = digits.target, cmap = plt.cm.get_cmap('cubehelix', 6))
plt.colorbar(ticks = range(0, 6), label = 'digit value')
plt.clim(-0.5, 5.5)

#%%
# Subplots
# plt.axes: Subplots by Hand
import numpy as np
import matplotlib.pyplot as plt
%matplotlib
plt.style.use('seaborn-white')

#%%
ax1 = plt.axes()    # standard axes
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])

# the equivlaent of this command withing he object oriented interface is
# fig.add_axes().
fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
                   xticklabels = [], ylim = (-1.2, 1.2))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
                   ylim = (-1.2, 1.2))

x = np.linspace(0, 10)
ax1.plot(np.sin(x))
ax2.plot(np.cos(x))

#%%
for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.text(0.5, 0.5, str((2, 3, i)),
             fontsize = 18, ha = 'center')

# te+be follwoing code, uses the equivalent obect-oriented command
fig = plt.figure()
fig.subplots_adjust(hspace = 0.4, wspace = 0.4)
for i in range(1, 7):
    ax = fig.add_subplot(2, 3, i)
    ax.text(0.5, 0.5, str((2, 3, i)),
            fontsize = 18, ha = 'center')
    
#%%
# plt.subplots: The Whole Grid in One Go
fig, ax = plt.subplots(2, 3, sharex = 'col', sharey = 'row')

# note that  by spcecifying sharex and sharey, we've automatically removed inner labels
# on the grid to make the plot cleaner

# axes are in a two-dimensional array, indexed by [row, col]
for i in range(2):
    for j in range(3):
        ax[i, j].text(0.5, 0.5, str((i, j)),
                      fontsize = 18, ha = 'center')
fig

#%%
# plt.GridSpec: More Complocated Arrangements
# the plt.GridSpec() object does not create a plot by itself; it is simply
# a convenient interface that is recognized by the plt.subplot() command
# for example, a gridspec for a grid of two rows and three columns with
# some specified width and height space looks like this
grid = plt.GridSpec(2, 3, wspace = 0.4, hspace = 0.3)

plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2])


# this ttype of flexible grid alignment has a wide range of uses
# create some normally distributed data
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 3000).T

# set yp the axes with fgridspec
fig = plt.figure(figsize = (6, 6))
grid = plt.GridSpec(4, 4, hspace = 0.2, wspace = 0.2)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels = [], sharey = main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels = [], sharex = main_ax)

# scatter points on the main axes
main_ax.plot(x, y, 'ok', markersize = 3, alpha = 0.2)

# histogram on the attached axes
x_hist.hist(x, 40, histtype = 'stepfilled',
            orientation = 'vertical', color = 'gray')
x_hist.invert_yaxis()

y_hist.hist(y, 40, histtype = 'stepfilled',
            orientation = 'horizontal', color = 'gray')
y_hist.invert_xaxis()



#%%
# text and Annotation
%matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd

births = pd.read_csv('data/births.csv')

quartiles = np.percentile(births['births'], [25, 50, 75])
mu, sigma = quartiles[1], 0.74 * (quartiles[2] - quartiles[0])
births = births.query('(births > @mu - 5 * @sigma) & (births < @mu + 5 * @sigma)')

births['day'] = births['day'].astype(int)

births.index = pd.to_datetime(10000 * births.year + 
                              100 * births.month +
                              births.day, format = '%Y%m%d')

births_by_date = births.pivot_table('births',
                                    [births.index.month, births.index.day])

births_by_date.index = [pd.datetime(2012, month, day)
                        for (month, day) in births_by_date.index]

fig, ax = plt.subplots(figsize = (12, 4))
births_by_date.plot(ax = ax)

#%%
fig, ax = plt.subplots(figsize = (12, 4))
births_by_date.plot(ax = ax)

# add labels to the plot
style = dict(size = 10, color = 'gray')

ax.text('2012-1-1', 3950, "New Year's Day", **style)
ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style)
ax.text('2012-9-4', 4850, "Labor Day", ha='center', **style)
ax.text('2012-10-31', 4600, "Halloween", ha='right', **style)
ax.text('2012-11-25', 4450, "Thanksgiving", ha='center', **style)
ax.text('2012-12-25', 3850, "Christmas ", ha='right', **style)

# Label the axes
ax.set(title='USA births by day of year (1969-1988)',
       ylabel='average daily births')

# format the x axis with centered month labels
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'))


#%%
# Arrows and Annotation
fig, ax = plt.subplots()

x = np.linspace(0, 20, 1000)
ax.plot(x, np.cos(x))
ax.axis('equal')

ax.annotate('local maximum', xy = (6.28, 1), xytext = (10, 4),
            arrowprops = dict(facecolor = 'black', shrink = 0.05))

ax.annotate('local minimum', xy = (5 * np.pi, -1), xytext = (2, -6),
            arrowprops = dict(arrowstyle = '->',
                              connectionstyle = 'angle3,angleA=0,angleB=-90'))

#%%
fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)
# Add labels to the plot
ax.annotate("New Year's Day", xy=('2012-1-1', 4100), xycoords='data',
            xytext=(50, -30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-0.2"))

ax.annotate("Independence Day", xy=('2012-7-4', 4250), xycoords='data',
            bbox=dict(boxstyle="round", fc="none", ec="gray"),
            xytext=(10, -40), textcoords='offset points', ha='center',
            arrowprops=dict(arrowstyle="->"))

ax.annotate('Labor Day', xy=('2012-9-4', 4850), xycoords='data', ha='center',
            xytext=(0, -20), textcoords='offset points')

ax.annotate('', xy=('2012-9-1', 4850), xytext=('2012-9-7', 4850),
            xycoords='data', textcoords='data',
            arrowprops={'arrowstyle': '|-|,widthA=0.2,widthB=0.2', })

ax.annotate('Halloween', xy=('2012-10-31', 4600), xycoords='data',
            xytext=(-80, -40), textcoords='offset points',
            arrowprops=dict(arrowstyle="fancy",
                            fc="0.6", ec="none",
                            connectionstyle="angle3,angleA=0,angleB=-90"))

ax.annotate('Thanksgiving', xy=('2012-11-25', 4500), xycoords='data',
            xytext=(-120, -60), textcoords='offset points',
            bbox=dict(boxstyle="round4,pad=.5", fc="0.9"),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle,angleA=0,angleB=80,rad=20"))

ax.annotate('Christmas', xy=('2012-12-25', 3850), xycoords='data',
            xytext=(-30, 0), textcoords='offset points',
            size=13, ha='right', va="center",
            bbox=dict(boxstyle="round", alpha=0.1),
            arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1));

# Label the axes
ax.set(title='USA births by day of year (1969-1988)',
       ylabel='average daily births')


# Format the x axis with centered month labels
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'));
ax.set_ylim(3600, 5400);

#%%


#%%
# three dimensional plotting in matplotlib
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection = '3d')


#%%
# Three dimensional points and lines
ax = plt.axes(projection = '3d')

# data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')

# data for three-dimensionalscattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c = zdata, cmap = 'Greens')

#%%
# three dimensional contour plots
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

#%%
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.contour3D(X, Y, Z, 50, cmap = 'binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.view_init(60, 35)

#%%
# WireFrames and SurfacePlots
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_wireframe(X, Y, Z, color = 'black')
ax.set_title('wireframe')

#%% surface
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1,
                cmap = 'viridis', edgecolor = 'none')
az.set_title('surface')

#%%
r = np.linspace(0, 6, 40)
theta = np.linspace(-0.9 * np.pi, 0.8 * np.pi, 20)
r, theta = np.meshgrid(r, theta)

X = r * np.sin(theta)
Y = r * np.cos(theta)
Z = f(X, Y)

ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1,
                cmap = 'viridis', edgecolor = 'none')



#%%
# Seaborn Histograms, KDEs and densities
import seaborn as sns
plt.figure()
data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size = 2000)
data = pd.DataFrame(data, columns = ['x', 'y'])

for col in 'xy':
    plt.hist(data[col], density = True, alpha = 0.5)
    
# rather than a histogram, we can get a smooth estimate pf the dostribution usong
    # a kern el density estimation
for col in 'xy':
    sns.kdeplot(data[col], shade = True)
plt.show()

#%%
sns.distplot(data['x'])
sns.distplot(data['y'])
plt.show()

#%%
with sns.axes_style('white'):
    sns.jointplot('x', 'y', data, kind = 'kde')
    
#%%
with sns.axes_style('white'):
    sns.jointplot('x', 'y', data, kind = 'hex')
    
    
    
#%% paitplots
iris = sns.load_dataset('iris')
iris.head()

#%%
sns.pairplot(iris, hue = 'species', size = 2.5)

#%%
tips = sns.load_dataset('tips')
tips.head()

tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']

grid = sns.FacetGrid(tips, row = 'sex', col = 'time', margin_titles = True)
grid.map(plt.hist, 'tip_pct', bins = np.linspace(0, 40, 15))

#%% factor plots
with sns.axes_style(style = 'ticks'):
    g = sns.factorplot('day', 'total_bill', 'sex', data = tips, kind = 'box')
    g.set_axis_labels('day', 'total bill')
    
#%% joint distributions
with sns.axes_style('white'):
    sns.jointplot('total_bill', 'tip', data = tips, kind = 'hex')

#%%
# kernel density estimation
sns.jointplot('total_bill', 'tip', data = tips, kind = 'reg')
plt.show()



#%%
# EXPLORING MARATHON FINISHING TIMES
%matplotlib
import pandas as pd
import numpy as np
from datetime import timedelta
data = pd.read_csv('data/marathon-data.csv')
data.head()

# dfix time format
def convert_time(s):
    h, m, s = map(int, s.split(':'))
    return timedelta(hours = h, minutes = m, seconds = s)

data = pd.read_csv('data/marathon-data.csv',
                   converters = {'split': convert_time, 'final': convert_time})

data.dtypes

data['split_sec'] = data['split'].astype(int) / 1E9
data['final_sec'] = data['final'].astype(int) / 1E9
data.head()

#%%
import seaborn as sns
import matplotlib.pyplot as plt
with sns.axes_style('white'):
    g = sns.jointplot('split_sec', 'final_sec', data, kind = 'hex')
    g.ax_joint.plot(np.linspace(4000, 16000),
                    np.linspace(8000, 32000), ':k')
#%%
data['split_frac'] = 1 - 2 * data['split_sec'] / data['final_sec']
data.head()
plt.figure()
sns.distplot(data['split_frac'], kde = False)
plt.axvline(0, color = 'k', linestyle = '--')
#%%
sum(data['split_frac'] < 0)
# out of nearly 40,000 participants, there were only 250 people who
# negative-split thir marathon
#%%
# lets see whether there is any correlation between this split fraction and
# other variables
g = sns.PairGrid(data, vars = ['age', 'split_sec', 'final_sec', 'split_frac'],
                 hue = 'gender', palette = 'RdBu_r')
g.map(plt.scatter, alpha = 0.78)
g.add_legend()
#%%
# It looks like the split fraction does not correlate particularly with age, but does corre‐
# late with the final time: faster runners tend to have closer to even splits on their mara‐
# thon time.
#%%
sns.kdeplot(data['split_frac'][data['gender'] == 'M'], label = 'men', shade = True)
sns.kdeplot(data['split_frac'][data['gender'] == 'W'], label = 'women', shade = True)
plt.xlabel('split_frac')

#%%
sns.violinplot('gender', 'split_frac', data = data,
               palette = ['lightblue', 'lightpink'])

'''
The interesting thing here is that there are many more men than women who are
running close to an even split! This almost looks like some kind of bimodal distribu‐
tion among the men and women. Let’s see if we can suss out what’s going on by look‐
ing at the distributions as a function of age.
'''
#%%
data['age_desc'] =data.age.map(lambda age: 10 * (age // 10))
data.head()

men = (data['gender'] == 'M')
women = (data['gender'] == 'W')
#%%
with sns.axes_style(style = None):
    sns.violinplot('age_desc', 'split_frac', hue = 'gender', data = data,
                   split = True, inner = 'quartile',
                   palette = ['lightblue', 'lightpink'])
#%%
plt.figure()
sns.kdeplot(data['age'][data['gender'] == 'M'], label = 'men', shade = True)
sns.kdeplot(data['age'][data['gender'] == 'W'], label = 'women', shade = True)
plt.axvline(25, color = 'k', linestyle = '--')

#%%
plt.figure()
sns.kdeplot(data['final_sec'][data['gender'] == 'M'], label = 'men', shade = True)
sns.kdeplot(data['final_sec'][data['gender'] == 'W'], label = 'women', shade = True)