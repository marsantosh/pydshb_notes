# -*- coding: utf-8 -*-
# chapter03.py
#

import numpy as np
import pandas as pd

# The pandas Series object
data = pd.Series([0.25, 0.5, 0.75, 1.0])
data

# get values from series
data.values

# get the index (array like object)
data.index


# unse strings as index
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index = ['a', 'b', 'c', 'd'])

data['b']

# noncontiguous indices
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index = [2, 5, 3, 7])
data
data[5]


# A Series is a struvtuere that maps typed keys to a set of typed values
population_dict = {
    'California': 8987899,
    'Texas': 56387565,
    'New York': 876485856,
    'Florida': 873857597,
    'Illinois': 9789797
}
population = pd.Series(population_dict)
population
population['California':'Illinois']

# specifying index
pd.Series(5, index = [100, 200, 300])

# data can be a dictionary, in which index defaults to the sorted
# dictionary keys
pd.Series({2:'a', 1:'b', 3:'c'})

# in each case, the index can be explicitly set if a different result is
# preferred
pd.Series({2:'a', 1:'b', 3:'c'}, index = [3, 2])

# DataFrame as a generalized NumPy array
area_dict = {
    'California': 34345,
    'Texas': 59596,
    'New York': 596886,
    'Florida': 77585,
    'Illinois': 254545
}
area = pd.Series(area_dict)
area


states = pd.DataFrame(
    {
     'population': population,
     'area': area
    }
)
states
states.index
states.columns

#%%
# Constructing DataFrame objects
#
# From a single Series object
pd.DataFrame(population, columns = ['population'])

# from a list of dicts
data = [{'a': i, 'b': 2 * i}
        for i in range(0, 3)]
pd.DataFrame(data)

# even if some keys in the dictionary are missing, Pandas will fill them in
# with NaN
pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])

# from a dictio anry of Series objects
pd.DataFrame({'population': population,
              'area': area})


# From a two-dimensional NumPy array
# Given a two-dimensional array of data, we can createa a DataFrame with any
# specified column and index names
# if omitted, an integer index will be used for each
pd.DataFrame(
    np.random.rand(3, 2),
    columns = ['foo', 'bar'],
    index = ['a', 'b', 'c']
)

# from nummpy structured array
A = np.zeros(3, dtype = [('A', 'i8'), ('B', 'f8')])
pd.DataFrame(A)


#%%
#
# The Pandas Index Object can be thought of either as an immutable array or as an
# ordered set 
ind = pd.Index([2, 3, 5, 7, 11])
ind
ind[:3]
print(ind.size, ind.shape, ind.ndim, ind.dtype)

# indexed cannot be modified bia normal means
ind[3] = 0


# index as ordered set
indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])

indA & indB # intersection
indA | indB # union
indA ^ indB # symmetric difference

# this operations may also be accessed via object mehtods
indA.intersection(indB)

#%%
#
# Series as dictionary
import pandas as pd
data = pd.Series([0.25, 0.50, 0.75, 1.0],
                 index = ['a', 'b', 'c', 'd'])
data
data['b']

# we can also use dictionary-like python expressions and methods to
# examine the key/indices and values
'a' in data
data.keys()
list(data.items())

data['e'] = 1.25
data

# Series as one-dimensional array
# slicing by explocit index
data['a':'c']
data[0:2]

# slicing by implicit index
data[0:2]

# masking
data[(data > 0.3) & (data < 0.8)]

# fancy indexing
data[['a', 'e']]

#%%
# Indexers: loc, iloc and ix
# These slicing and indexing conventions can be a source of confusion
# For example, if your Series has an explicit integer index, an indexing operation
# such as data[1] will use the explicit indices, while a slicing operation like
# data[1:3] will use the implicit Python-style index
data = pd.Series(['a', 'b', 'c'], index = [1, 3, 5])
data
data[1]
data[1:3]

# because of this potential confusion in the case of integer indexed,
# Pandas provides some special indexer attributes that explicitly expose
# certain indexing schemes
# These arae not funcitonal methods, but atributes that expose
# a particular slicing interface to the data in the Series
#
# First the loc attribute allows indexing and slicing that always references
# the explicit index
data.loc[1]
data.loc[1:3]
#
# The iloc attribute allows indexing and slicing that always references the
# implcit Python-style index:
data.iloc[1]
data.iloc[1:3]

#%%
# DataFrame as a dictionary
#
area = pd.Series({'California': 423967, 'Texas': 695662,
        'New York': 141297, 'Florida': 170312,
        'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
        'New York': 19651127, 'Florida': 19552860,
        'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
data

data['area']
data.area
data.area is data['area']


# you should avoid the temptation to try column assignment via
# attribute
# (i.e. use data['pop'] = z rather than data.pop = z)

data['density'] = data['pop'] / data['area']
data

# values attribute
data.values

# Transpose
data.T

# to get row access:
data.values[0]
data.values[0:3]

# in DataFrame, using the iloc indexer, we can index
# the underlying array as if it is a simple NumPy array (using the implicit
# Python style index), but the DataFrame index and column labels are maintained
# in the result
data.iloc[:3, :2]
data.loc[:'Illinois', :'pop']

# the ix indexer allows a hybrid of these two approaches
data.ix[:3, :'pop']

data.loc[data.density > 100, ['pop', 'density']]
data.iloc[0, 2] = 90
data


#%%
#
# Operating on Data in Pandas
#
# Ufuncs: Idbnex preservation
import numpy as np
import pandas as pd

rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4))
ser

df = pd.DataFrame(rng.randint(0, 10, (3, 4)),
                  columns = ['A', 'B', 'C', 'D'])

df

# If we apply a NumPy ufunc on either of these objects, the result will be another
# Pandas pbject with the indices preseved
np.exp(ser)

np.sin(df * np.pi / 4)

#%% Ufuncs: Index alignment
#
# For binary operations on two Series or dataFrame opbjects, Pandas will
# align indices in the process of performing the operation
area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
                  'California': 423967}, name='area')
population = pd.Series({'California': 38332521, 'Texas': 26448193,
                        'New York': 19651127}, name='population')

population / area

A = pd.Series([2, 4, 6], index = [0, 1, 2])
B = pd.Series([1, 3, 5], index = [1, 2, 3])
A + B

A.add(B, fill_value = 0)

# Index alignment in DataFrame
#
# A similar type of alignment takes place for both column and indices when
# youare performing operations on DataFrames
A = pd.DataFrame(rng.randint(0, 20, (2, 2)),
                 columns = list('AB'))
A
B = pd.DataFrame(rng.randint(0, 10, (3, 3)),
                 columns = list('BAC'))
B
A + B

fill = A.stack().mean()
A.add(B, fill_value = fill)

#%% Ufuncs: Operations between DataFrame and Series
# Operations between a DataFrame and a Series are similar to operations
# between a two-dimensional and one-dimensional NumPy array
A = rng.randint(10, size = (3, 4))
A
A - A[0]

# According to NumPy's broadcasting rules, subtraction between a two-dimensional
# array and one of its rows is applied row-wise
df = pd.DataFrame(A, columns = list('QRST'))
df - df.iloc[0]

# if you would instead like to operate column-wise, yuou can use the
# object methods mentioned earlier, while specifying the axis keyword
df.subtract(df['R'], axis = 0)

# Note that these DataFrame/Series operations, like the operations discussed before,
# will automatically align indices between the two elements
halfrow = df.iloc[0, ::2]
halfrow
df - halfrow



#%% Missing Data in Pandas
#
# None: Pythonid missing data
import numpy as np
import pandas as pd

vals1 = np.array([1, None, 3, 4])
vals1
vals2 = np.array([1, np.nan, 3, 4])
vals2.dtype

# upcasting
x = pd.Series(range(0, 2), dtype = int)
x
x[0] = None
x

#%% Operating on null values
data = pd.Series([1, np.nan, 'hello', None])
data.isnull()
data[data.notnull()]

data.dropna()
df = pd.DataFrame([[1, np.nan, 2],
                   [2, 3, 5],
                   [np.nan, 4, 6]])

df
df.dropna()
df.dropna(axis = 1)
df[3] = np.nan
df
df.dropna(axis = 1, how = 'all')

df.dropna(axis = 0, thresh = 3)

data = pd.Series([1, np.nan, 2, None, 3], index = list('acbde'))
data
data.fillna(0)

# forward fill
data.fillna(method = 'ffill')

# nbackfill
data.fillna(method = 'bfill')

df.fillna(method = 'ffill', axis = 1)

#%%
# Hierarchical Indexing
import numpy as np
import pandas as pd

index = [
    ('California', 2000),
    ('California', 2010),
    ('New York', 2000),
    ('New York', 2010),
    ('Texas', 2000),
    ('Texas', 2010)
]
populations = [33871648, 37253956,
    18976457, 19378102,
    20851820, 25145561
]
pop = pd.Series(populations, index = index)
pop

# The MultiIndex way
index = pd.MultiIndex.from_tuples(index)
index
pop = pop.reindex(index)
pop
pop[:, 2010]

# Multiindex as extra dimension
# The unstack() method will quickly convert a multiply-indexed
# Series into a conventionally indexed DataFrame
pop_df = pop.unstack()
pop_df
pop_df.stack()

pop_df = pd.DataFrame({'total': pop,
                       'under18': [9267089, 9284094,
                                4687374, 4318033,
                                5906301, 6879014]})
pop_df
f_u18 = pop_df['under18'] / pop_df['total']
f_u18.unstack()
 
# Methods MultiIndex creation
df = pd.DataFrame(np.random.rand(4, 2),
                  index = [['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns = ['data1', 'data2'])
df

# Explicit multiIndex constrcutors
pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])

# you can even construct it from a Cartesian product of single indices:
pd.MultiIndex.from_product([['a', 'b'], [1, 2]])

pd.MultiIndex(levels = [['a', 'b'], [1, 2]],
              codes = [[0, 0, 1, 1], [0, 1, 0, 1]])


# MultiIndex level names
# Sometimes it is convenient to name the leves of the MultiIndex
pop.index.names = ['state', 'year']
pop


# MultiIndex for columns
# hierarchical indices and columns
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names = ['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'],['HR', 'Temp']],
                                     names = ['subject', 'type'])

# mock some data
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37

# create the dataframe
health_data = pd.DataFrame(data, index = index, columns = columns)
health_data
health_data['Guido']


pop
pop['California', 2000]

pop[pop > 22000000]

pop[['California', 'Texas']]

health_data

health_data['Guido','HR']
health_data.iloc[:2, :2]
health_data.loc[:, ('Bob', 'HR')]


idz = pd.IndexSlice
health_data.loc[idz[:, 1], idz[:, 'HR']]

#%%
#
# Re arranging Mulyi-Indices
#
index = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
data = pd.Series(np.random.randn(6), index = index)
data.indeex.names = ['char', 'int']
data

# error for partial slicing for non sorted multiindex
try:
    data['a':'b']
except KeyError as e:
    print(type(e))
    print(e)
    
data = data.sort_index()
data

# now, partial slicing will work as expected
try:
    print(data['a':'b'])
except KeyError as e:
    print(type(e))
    print(e)
    
#%%
# Stacking and unstacking indices
pop
pop.unstack(level = 0)
pop.unstack(level = 1)

pop.unstack().stack()

#%% 
# Index ssetting and resetting
#
pop_flat = pop.reset_index(name = 'population')
pop_flat

pop_flat.set_index(['state', 'year'])

data_mean = health_data.mean(level = 'year')
data_mean

# by firther making use of the axis keyword, we can take the man among levels
# on the columns as well
data_mean.mean(axis = 1, level = 'type')


#%%
#
# Combining DataSets: Concat and Append
import numpy as np
import pandas as pd

def make_df(cols, ind):
    '''Quickly make a DataFrame'''
    data = {c: [str(c) + str(i) for i in ind]
            for c in cols}
    return pd.DataFrame(data, ind)

# make a dataframe
make_df('ABCDER', range(0, 9))

# racall concatenate numpy
x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
np.concatenate([x, y, z])

x = [[1, 2],
     [3, 4]]
np.concatenate([x, x], axis = 1)


# pd.concat() can be used for  asimple concatenation of Series or DataFrames
# objects just as np.concatenate() can be used for simple concatenations of arrays
ser1 = pd.Series(['A', 'B', 'C'], index = [1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index = [4, 5, 6])
ser1
ser2
pd.concat([ser1, ser2])

# it also works to concatenate highter-dimensional objects, such as DataFrames
df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
print(df1)
print(df2)
print(pd.concat([df1, df2]))

df3 = make_df('AB', [0, 1])
df4 = make_df('CD', [0, 1])
print(df3)
print(df4)
print(pd.concat([df3, df4], axis = 1))

#%%
# duplicate indices
# One important difference between np.concatenate and pd.concat is that Pandas
# concatenation preserves indices, een if the result will have duplicate indices
x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])

y.index = x.index    # make duplicate indices
print(x); print(y); print(pd.concat([x, y]))


# Catching the repeats as an error
try:
    pd.concat([x, y], verify_integrity = True)
except ValueError as e:
    print('ValueError:', e)
    
# ignore index
print(x); print(y); print(pd.concat([x, y], ignore_index = True))

# adding multiIndex keys
# Another alternative is to use the keys option to specify a label for the
# data sources; the result will be a hierarchically indexed series containing
# the data
print(x); print(y); print(pd.concat([x, y], keys = ['x', 'y']))


# append method
df1.append(df2)

#%%
pop = pd.read_csv('data/state-population.csv')
areas = pd.read_csv('data/state-areas.csv')
abbrevs = pd.read_csv('data/state-abbrevs.csv')

print(pop.head())
print(areas.head())
print(abbrevs.head())

#%%
merged = pd.merge(pop, abbrevs, how = 'outer',
                  left_on = 'state/region', right_on = 'abbreviation')
merged = merged.drop('abbreviation', 1)
merged.head()

#%%
# check for missmatches
merged.isnull().any()

# some of the population info is null; lets configure which these are
merged[merged['population'].isnull()].head()

#%%
merged.loc[merged['state'].isnull(), 'state/region'].unique()

merged.loc[merged['state/region'] == 'PR', 'state'] = 'Puerto Rico'
merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'
merged.isnull().any()

#%% merging data
final = pd.merge(merged, areas, on = 'state', how = 'left')
final.head()

#%% checking for nnulls
final.isnull().any()
final['state'][final['area (sq. mi)'].isnull()].unique()
final.dropna(inplace = True)
final.head()

#%%
data2010 = final.query('year == 2010 & ages == "total"')
data2010.head()

#%% compute population density and display it in order
data2010.set_index('state', inplace = True)
density = data2010['population'] / data2010['area (sq. mi)']
density.sort_values(ascending = False, inplace = True)
density

#%%
# Aggregation and Grouping
import seaborn as sns
planets = sns.load_dataset('planets')
planets.shape
planets.head()

#%%
planets.dropna().describe() 

#%%
planets.groupby('method')
#%%
planets.groupby('method')['orbital_period']
#%%
# This gives an idea of the general scale of orbital periods )in days)
# that each method is sensitive to
planets.groupby('method')['orbital_period'].median()

#%%
# iteration over groups
# The GroupBy object supports direct iteration over the groups,
# returning each group as a Series or dataFrame
for (method, group) in planets.groupby('method'):
    print('{0:30s} shape = {1}'.format(method, group.shape))
    
#%%
planets.groupby('method')['year'].describe().unstack()

#%%
# Aggregate, filter, transform, apply
rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(0, 6),
                   'data2': rng.randint(0, 10, 6)},
                    columns = ['key', 'data1', 'data2'])
df

#%%
# aggregation
# can compute different aggregates at once
df.groupby('key').aggregate(['min', np.median, max])

# another useful patern is to pass a dictionaty mapping column names to
# operations to be applied on that column
df.groupby('key').aggregate({'data1': 'min',
                             'data2': 'max'})
#%%
# filtering
# A filtering operation allows you to drop data based on the group
# properties
def filter_func(x):
    return x['data2'].std() > 4

print(df)
print(df.groupby('key').std())
print(df.groupby('key').filter(filter_func))

# The filter() function should return a boolean value specifying
# whether the group passes the filtering

#%%
# transformation
# While aggregation must return a reduced version of the data,
# transformation can retrun some transformed version of the full data
# to recombine.
# For such a transformtion, the output is the same shape as the input
# A common example is to center the data by subtracting the group-wise
# mean
df.groupby('key').transform(lambda x: x - x.mean())

#%%
# apply
# The apply() method lets you apply an arbitrary function to the
# group results. The function should take a DataFrame, and return either a 
# Pandas object or a scalar;; the combine operation will be tailored
 # to the type of output returned
def norm_by_data2(x):
    # x is a DataFrame of group values
    x['data1'] /= x['data2'].sum()
    return x

print(df)
print(df.groupby('key').apply(norm_by_data2))

#%%
# A dictionary or series mapping index to group
df2 = df.set_index('key')
mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
print(df2)
print(df2.groupby(mapping).sum())

# Any python function
print(df2.groupby(str.lower).mean())

# a list of valid keys,
# any of the precided choices can be combined to grou p on a multi-index
print(df2.groupby([str.lower, mapping]).mean())

#%%
# example
decade = 10 * (planets['year'] // 10)
decade = decade.astype(str) + 's'
decade.name = 'decade'
planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)

#%%
planets
#%%
planets.groupby(['method', decade])['number', 'distance'].sum()

#%%
# Motivating Puvot Tables
import numpy as np
import pandas as pd
import seaborn as sns
titanic = sns.load_dataset('titanic')
titanic.head()

#%%
# survival rate by gender
titanic.groupby('sex')[['survived']].mean()

#%%
# by gender and class
titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()

#%%
# pivot table syntax
print(titanic.pivot_table('survived', index = 'sex', columns = 'class', aggfunc = 'count'))
print(titanic.pivot_table('survived', index = 'sex', columns = 'class', aggfunc = 'mean'))

#%%
# multileel pivot tables
age = pd.cut(titanic['age'], [0, 18, 80])
print(titanic.pivot_table('survived', ['sex', age], 'class', aggfunc = 'count'))
print()
print(titanic.pivot_table('survived', ['sex', age], 'class', aggfunc = 'mean'))

#%%
fare = pd.qcut(titanic['fare'], 2)
titanic.pivot_table('survived', ['sex', age], [fare, 'class'])

#%%
titanic.pivot_table(index = 'sex',
                    columns = 'class',
                    aggfunc = {'survived': sum, 'fare': 'mean'})

#%%
# at times it's useful to compute totals along each grouping
titanic.pivot_table('survived',
                    index = 'sex',
                    columns = 'class',
                    margins = True)
# here this automatically gives us information about the class-agnostic
# survival rate by gender, the gender-agnostic survival rate by class,
# and the overall survival rate of 38%

#%%
# example: birthrates
births = pd.read_csv('data/births.csv')
births.head()

#%%
births['decade'] = 10 * (births['year'] // 10)
births.pivot_table('births', index = 'gender', columns = 'decade',
                   aggfunc = 'sum')

#%%
%matplotlib
import matplotlib.pyplot as plt
sns.set()
births.pivot_table('births', index = 'year', columns = 'gender',
                   aggfunc = 'sum').plot()
plt.ylabel('total births per year')

#%%
# sigma clipping
quartiles = np.percentile(births['births'], [25, 50, 75])
mu = quartiles[1]
sigma = 0.74 * (quartiles[2] - quartiles[0])

#%%
births = births.query('(births > @mu - 5 * @sigma) & (births < @mu + 5 * @sigma)')

#%%
# set day column to integer, it originally was a string due to nulls
births.loc['day'] = births['day'].astype(int)

#%%
# create a datetime index from the year, month, day
births.index = pd.to_datetime(10000 * births.year +
                              100 * births.month +
                              births.day, format = '%Y%m%d')
births['dayofweek'] = births.index.dayofweek
#%%
births.pivot_table('births',
                   index = 'dayofweek',
                   columns = 'decade',
                   aggfunc = 'mean').plot()
plt.gca().set_xticklabels(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
plt.ylabel('mean births by day')

# %%
# plot the mean number of births bu the dat of the year
births_by_date = births.pivot_table('births',
                                    [births.index.month, births.index.day])
births_by_date.index
births_by_date.head()

# the result is a multi-iondex over months and days
# to make this easilt plottable, let's turn these months and days
# into a date by associating them with a dummy year variable

#%%
births_by_date.index = [pd.datetime(2012, int(month), int(day))
                        for (month, day) in births_by_date.index]
births_by_date.head()

#%%
# plot the results
fig, ax = plt.subplots(figsize = (12, 4))
births_by_date.plot(ax = ax)
#
#In particular, the striking feature of this graph is the dip in birthrate on US holidays
#(e.g., Independence Day, Labor Day, Thanksgiving, Christmas, New Year’s Day)
#although this likely reflects trends in scheduled/induced births rather than some deep
#psychosomatic effect on natural births.

#%%
# Recipe Database
try:
    recipes = pd.read_json('data/recipeitems-latest.json')
except ValueError as e:
    print('ValueError: ', e)
    
#%%
with open('data/recipeitems-latest.json') as f:
    line = f.readline()
pd.read_json(line).shape

#%%
# read the entire file into a Python array
with open('data/recipeitems-latest.json', 'r') as f:
    # extract each line
    data = (line.strip() for line in f)
    # reformat so eacg line is the element of a list
    data_json = '[{0}]'.format(','.join(data))
# readthe resylt as a JSON
recipes = pd.read_json(data_json)
recipes.shape

#%%
recipes.iloc[0]

#%%
recipes.ingredients.str.len().describe()

#%%
# recipe with the longest ingredient list
recipes.name[np.argmax(recipes.ingredients.str.len())]

#%%
# how many recipes are for breakfast food
recipes.description.str.contains('[Bb]reakfast').sum()

#%%
# how many of the recipes list cinnamon as an ingredient
recipes.ingredients.str.contains('[Cc]innamon').sum()

#%%
# we could even look to see whether any recipes misspell the ingredient
# as 'cinamon'
recipes.ingredients.str.contains('[Cc]inamon').sum()

#%%
# A simple recipe recommender
#
spice_list = ['salt', 'pepper', 'oregano', 'sage', 'parsley',
              'rosemary', 'tarragon', 'thyme', 'paprika', 'cumin']

#%%
import re
spice_df = pd.DataFrame(
    dict((spice, recipes.ingredients.str.contains(spice, re.IGNORECASE))
    for spice in spice_list)
)
spice_df.head()

#%%
# get recipe that uses parsley, paprika and tarragon
selection = spice_df.query('parsley & paprika & tarragon')
len(selection)

#%%
recipes.name[selection.index]

#%%
# Reshampling, Shifting and Windowing
#
from pandas_datareader import data
goog = data.DataReader('GOOG', 'yahoo', start = '2004', end = '2018')
goog.head()

#%%
# using closing price
goog = goog['Close']

#%%
%matplotlib
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

#%%
goog.plot()

#%%
# Resampling and converting frequencies
goog.plot(alpha = 0.5, style = '-')
goog.resample('BA').mean().plot(style = ':')
goog.asfreq('BA').plot(style = '--')
plt.legend(['input', 'resample', 'asfreq'],
           loc = 'upper left')

#%%
# Resampling the business day data at a daily frequence
fig, ax = plt.subplots(2, sharex = True)
data = goog.iloc[:10]

data.asfreq('D').plot(ax = ax[0], marker = 'o')

data.asfreq('D', method = 'bfill').plot(ax = ax[1], style = '-o')
data.asfreq('D', method = 'ffill').plot(ax = ax[1], style = '--o')
ax[1].legend(['back-fill', 'forward-fill'])


#%%
# Time-Shifts
fig, ax = plt.subplots(3, sharey = True)

# apply a frequency to the daya
goog = goog.asfreq('D', method = 'pad')

goog.plot(ax = ax[0])
goog.shift(900).plot(ax = ax[1])
goog.tshift(900).plot(ax = ax[2])

# legends and annotations
local_max = pd.to_datetime('2007-11-05')
offset = pd.Timedelta(900, 'D')

ax[0].legend(['input'], loc = 2)
ax[0].get_xticklabels()[4].set(weight = 'heavy', color = 'red')
ax[0].axvline(local_max, alpha = 0.3, color = 'red')

ax[1].legend(['shift(900)'], loc=2)
ax[1].get_xticklabels()[4].set(weight='heavy', color='red')
ax[1].axvline(local_max + offset, alpha=0.3, color='red')

ax[2].legend(['tshift(900)'], loc=2)
ax[2].get_xticklabels()[1].set(weight='heavy', color='red')
ax[2].axvline(local_max + offset, alpha=0.3, color='red')

#%%
# Rolling Windows
rolling = goog.rolling(365, center = True)
data = pd.DataFrame({'input': goog,
                     'one-year rolling_mean': rolling.mean(),
                     'one-year rolling_std': rolling.std()})
ax = data.plot(style = ['-', '--', ':'])
ax.lines[0].set_alpha(0.3)


#%%
# example: Visualizing Seattle Bycicle Counts
data = pd.read_csv('data/Fremont_Bridge.csv', index_col = 'Date',
                   parse_dates =True)

#%%
print(data.head())
print(data.columns)
#%%
data.columns = ['Total', 'East', 'West']

#%%
data.dropna().describe()

#%%
# Visualizing the data
data.plot()
plt.ylabel('Hourly Bicycle Count')

#%% resample data to a coarser grid
weekly = data.resample('W').sum()
weekly.plot(style = [':', '--', '-'])
plt.ylabel('Weekly bicycle count')

#%%
# aggregating data with rolling mean
daily = data.resample('D').sum()
daily.rolling(30, center = True).sum().plot(style = [':', '--', '-'])
plt.ylabel('mean hourly count')

#%%
# Ussing Gaussian window
daily.rolling(50, center = True,
              win_type = 'gaussian').sum(std = 10).plot()

#%%
# Digging in
# average traffic as a function of tge time of day
by_time = data.groupby(data.index.time).mean()
hourly_ticks = 4 * 60 * 60 * np.arange(6)
by_time.plot(xticks = hourly_ticks)

#%% 
# How things change based on the day of the week
by_weekday = data.groupby(data.index.dayofweek).mean()
by_weekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
by_weekday.plot()

#%%
# let's do a compound groupby and look at the hourly trend on
# weekdats versus weekends
weekend = np.where(data.index.weekday < 5, 'Weekday', 'Weekend')
by_time = data.groupby([weekend, data.index.time]).mean()

# plotting
fig, ax = plt.subplots(1, 2, figsize = (14, 5))
by_time.ix['Weekday'].plot(ax = ax[0], title = 'Weekdays',
                           xticks = hourly_ticks, style = [':', '--', '-'])
by_time.ix['Weekend'].plot(ax = ax[1] ,title = 'Weekends',
                           xticks = hourly_ticks, style = [':', '--' ,'-'])