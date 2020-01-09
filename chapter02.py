# -*- coding: utf-8 -*-
# chapter02.py
# introduction_to_numpys

import numpy as np
np.__version__

# fixed-type arrays in Python
import array
L = list(range(0, 10))
A = array.array('i', L)     # Here 'i' is a type code indicating the contents are integers.
A

# Much more useful, however, is the ndarray object of the NumPy package. While
# Python’s array object provides efficient storage of array-based data, NumPy adds to
# this efficient operations on that data. We will explore these operations in later sec‐
# tions; here we’ll demonstrate several ways of creating a NumPy array.
import numpy as np
np.array([1, 2, 3, 4, 5])

np.array([3.14, 4, 2, 3])


# If we want to explicityle set the data type of the resulting array,
# we can use the dtype keyword
np.array([1, 2, 3, 4], dtype='float32')

# nested lists result in multidimensional arrays
np.array([range(i, i + 3) for i in [2, 3, 4]])

# the inner lists are treated as rows of the resulting two-dimensional array

# create a length-10 integer array filled with zeros
np.zeros(10, dtype = int)

# create a 3 * 5 floating-point array filled with 1s
np.ones((3, 5), dtype = float)

# create a 3 x 5 array filled with 3.14
np.full((3, 5), 3.14)


# create an array filled with a linear sequence
# starting at 0, ending at 20, stepping by 2
# (this is similar to the built-in range() function)
np.arange(0, 20, 2)

# create an array of five values evenly spaced between 0 and 1
np.linspace(0, 1, 5)

# create a 3x3 array of ubniformlu distributed random
# values between 0 and 1
np.random.random((3, 3))

# create a 3x3 array of normally distributed random values with
# mean 0 and stddev 1
np.random.normal(0, 1, (3, 3))

# create a 3x3 identity matrix
np.eye(3)


#%%
import numpy as np
from pprint import pprint
np.random.seed(0)

x1 = np.random.randint(10, size = 6)
x2 = np.random.randint(10, size = (3, 4))
x3 = np.random.randint(10, size = (3, 4, 5))

print('x3 ndim: ', x3.ndim)
print('x3 shape: ', x3.shape)
print('x3 size: ', x3.size)
pprint(x3)

#%%
# other useful attributes include:
print('itemsize: ', x3.itemsize, 'bytes')
print('nbytes: ', x3.nbytes, 'bytes')

#%%
# Multidimensional subarrays
print(x2)
print(x2[1::2])
print(x2[::-1])   # all elements reversed
print(x2[5::-2])  # reversed evey other from index 5

# finally, subarray dimensions acan be reversed together:
print(x2[::-1, ::-1])


#%%
# Accesing array rows and columns
# access first column of x2
print(x2[:, 0])

# access first row of x2
print(x2[0, :])

# in the case of row access, the empty slice can beommited for a more
# compact syntax
print(x2[0])

#%%
# subarrays as no-copy views
# One important—and extremely useful—thing to know about array slices is that they
# return views rather than copies of the array data. This is one area in which NumPy
# array slicing differs from Python list slicing: in lists, slices will be copies. Consider our
# two-dimensional array from before:
pprint(x2)
x2_sub = x2[:2, :2]
pprint(x2_sub)

x2_sub[0, 0] = 99
pprint(x2_sub)

#%%
# to cpy an array
x2_sub_copy = x2[:2, :2].copy()
pprint(x2_sub_copy)
x2_sub_copy[0, 0] = 42
pprint(x2_sub_copy)
pprint(x2)

#%%
# Another useful type of operation is reshaping of arrays
# the most flexible way of doing this is with the
# reshape() method
grid = np.arange(1, 10).reshape((3, 3))
pprint(grid)

# another common reshaping pattern is the conversion of a one-dimensional array
# into a two-dimensional row or column matrix. You can do this with the reshape
# method, or more easily by making use of the newaxis keyword within a slice
# operation
x = np.array([1, 2, 3])

# row vector via reshape
x.reshape((1, 3))

# row vector via newaxis
x[np.newaxis, :]

# column vector via reshape
x.reshape((3, 1))

# column vector via newaxis
x[:, np.newaxis]

#%%
# Array concatenation and splitting
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])

# concatenate two arrays
pprint(np.concatenate([x, y]))

# concatenate more than two arrays
z = [99, 99, 99]
pprint(np.concatenate([x, y, z]))

# np.concatatenate can also be sued for two-dimensional arrays
grid = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# concatenate along the first axis:
pprint(np.concatenate([grid, grid]))

# concatenate alonog the second axis
pprint(np.concatenate([grid, grid], axis = 1))


#%%
# for working with arrays of mixed dimensions, it can be clearer
# to sue the np.vstack and np.hstack functions
x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])

# vertically stack the arrays
pprint(np.vstack([x, grid]))

# horizontally stack the arrays
y = np.array([[99],
              [99]])
pprint(np.hstack([grid, y]))

# similarly, np.dstack will stack arrays along the third axis

#%%
# splitting of arrays
# The opposite of concatenation is splitting
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)

# notice that N split points lead to N+1 subarrays.
# The related functions np.hsplit and np.vsplit are similar
grid = np.arange(16).reshape((4, 4))
print(grid, "\n")

upper, lower = np.vsplit(grid, [2])
print(upper, "\n")
print(lower, "\n")

left, right = np.hsplit(grid, [2])
print(left, "\n")
print(right, "\n")


#%% computations on large arrays
import numpy as np
np.random.seed(0)

def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output

values = np.random.randint(1, 10, size = 5)
compute_reciprocals(values)

# see computation in big_array
big_array = np.random.randint(1, 100, size = 1000000)
%timeit compute_reciprocals(big_array)

# It takes several seconds to compute these million operations and to store the result!
# When even cell phones have processing speeds measured in Giga-FLOPS (i.e., bil‐
# lions of numerical operations per second), this seems almost absurdly slow. It turns
# out that the bottleneck here is not the operations themselves, but the type-checking
# and function dispatches that CPython must do at each cycle of the loop. Each time
# the reciprocal is computed, Python first examines the object’s type and does a
# dynamic lookup of the correct function to use for that type. If we were working in
# compiled code instead, this type specification would be known before the code exe‐
# cutes and the result could be computed much more efficiently.


#%% vectorized computations
print(compute_reciprocals(values))
print(1.0 / values)

%timeit compute_reciprocals(big_array)
%timeit (1.0 / big_array)


#%%
# specialized ufuncs
from scipy import special

x = [1, 5, 10]
print('gamma(x)     : ', special.gamma(x))
print('ln|gamma(x)  : ', special.gammaln(x))
print('beta(x, 2)   : ', special.beta(x, 2))

# error function(integral of Gaussian)
# its complement, and its inverse
x = np.array([0, 0.3, 0.7, 1.0])
print("erf(x)       : ", special.erf(x))
print("erfc(x)      : ", special.erfc(x))
print("erfinv(x)    : ", special.erfinv(x))

#%%
# Advanced UFunc Features
# 
# Specifying output
# For large calcualtions, it is sometimes useful to be able to specify the
# array where the result of the calculation will be stored
# rather than creating a temporary array, you can use this to wrote computation
# results dirwctly to the memory location where you'd
# like thenm to be
# for all ufuncs, you can do this using the out argument of the function
x = np.arange(5000000)
y = np.empty(5000000)

np.multiply(x, 10, out = y)
print(y)

# can even be sued with array views
y = np.zeros(5000000 * 2)
np.power(2, x, out = y[::2])
print(y)

# If we had instead written y[::2] = 2 ** x , this would have resulted in the creation
# of a temporary array to hold the results of 2 ** x , followed by a second operation
# copying those values into the y array. This doesn’t make much of a difference for such
# a small computation, but for very large arrays the memory savings from careful use of
# the out argument can be significant.

#%%
# Aggregates
import numpy as np
x = np.arange(1, 6)

np.add.reduce(x)
np.multiply.reduce(x)
np.add.accumulate(x)
np.multiply.accumulate(x)

#%%
# Outer products
# Any ufun can compute the output of all pairs of two different inputs
# using the outer method
# This allows you, in one line, to do things like create a multiplication
# table
x = np.arange(1, 13)
np.multiply.outer(x, x)




#%%
# Summing the Values in an Array
# As a quick example, consider computing the sum of all values in an array
import numpy as np
L = np.random.random(100)
sum(L)
np.sum(L)

big_array = np.random.rand(1000000)
%timeit sum(big_array)
%timeit np.sum(big_array)

%timeit min(big_array)
%timeit np.min(big_array)

# for min, max, sum, and several other numpy aggregates, a shorter syntax
# is to use methods of the array object itself
print(big_array.min(), big_array.max(), big_array.sum())


#%%
# Multidimensional aggregates
# One common type of aggregation operation is an aggregate along a row or
# column
# Say you have some data stored in a two-dimensional array
import numpy as np
M = np.random.random((3, 4))
print(M)

# by default, each  numpy aggregation function will return the aggregate
# over the entire array
M.sum()

# aggregation fucntions take an additional argument specifying the axis along
# which the aggreagate is computed
# For example, we can find the minimum value within each column
# by specifying axis = 0
M.min(axis = 0)

# the function returns four values, corresponding to the four columns of
# numbrs
M.max(axis = 1)

# TheThe way the axis is specified here can be confusing to users coming from other lan‐
# guages. The axis keyword specifies the dimension of the array that will be collapsed,
# rather than the dimension that will be returned. So specifying axis=0 means that the way the axis is specified here can be confusing to users coming from other lan‐
# guages. The axis keyword specifies the dimension of the array that will be collapsed,
# rather than the dimension that will be returned. So specifying axis=0 means that the
# first axis will be collapsed: for two-dimensional arrays, this means that values within
# each column will be aggregated.

#%%
# What is the Average height of US Presidents?
!head -4 data/president_heights.csv

import pandas as pd
data = pd.read_csv('data/president_heights.csv')
heights = np.array(data['height(cm)'])
print(heights)

 # compute summary statistics
 print('Mean heights:     ', heights.mean())
 print('Std deviation:    ', heights.std())
 print('Minimum height:   ', heights.min())
 print('Maximum heights:  ', heights.max())

# the aggregation operation rediced the entire array to a sibgle
 # summarizing value
print('25th percentile:    ', np.percentile(heights, 25))
print('Median:    ', np.median(heights))
print('75th percentile:    ', np.percentile(heights, 75))

%matplotlib
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

plt.hist(heights)
plt.title('Height distribution of US Presidents')
plt.xlabel('height (cm)')
plt.ylabel('number')

#%%
# Computaiton on Arrays: Broadcasting
# We saw in the previous section how NumPy's universal functions can be used
# to vectorize operations and thereby remove slow Python loops
# Another means of vectorizing operations is to use NumPy's broadcasting
# functionality. Broadcasting is simply a set of rules for applying binary
# ufuncs on arrays of different sizes
#
# Introducing Broadcasting
import numpy as np
a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
a + b

a + 5

# Observe the result when we add a one-dimensipnal array to a two-dimensional
# array
M = np.ones((3, 3))
M

M + a
# Here the one-dimensional array a is stretched, or breadcast, across the
# second dimension in order to match the shape of M

a = np.arange(3)
b = np.arange(3)[:, np.newaxis]
print(a)
print(b)
a + b
# Here we stretched both a and b to matvch a common shape
# and the result is a two dimensional array

# !!!
# Rules of Broadcasting
# Broadcasting in NumPy follows a strict set of rules to determine the
# interaction between the two arrays
#
# Rule1: If the two arrays differ in their number of fimensions, the
# shape of the one with fewer dimensions is padded with ones on its
# leading (left) side
#
# Rule2: If the shape fo the two arrays does not match in any dimension, the
# array with shape equal to 1 in that dimension is stretched to match
# the other shape
#
# Rule3: If in any dimension the sized disagree and neither is equal to 1,
# an error is raised
# !!!
#
# Broadcasting example 1:
# adding a two-dimensional array to a one-dimensional array
M = np.ones((2, 3))
a = np.arange(3)
M.shape
a.shape
# we see by rule 1 that the array `a` has fewer dimensions, so we pad
# it on the left with ones
# M.shape --> (2, 3)
# a.shape --> (1, 3)
# By rule 2, we now see that the first dimension disagrees, so we stretch
# this dimension to match
# M.shape --> (2, 3)
# a.shape --> (2, 3)
# The shapes match, and we see that the final shape will be (2, 3)
M + a
(M + a).shape

# Broadacsting example2:
# Lets take a look at an example where both arrays need to be
# broadcast:
a = np.arange(3).reshape((3, 1))
b = np.arange(3)
a.shape
b.shape
# Rule 1 says we must pad the shape of b with ones
# a.shape --> (3, 1)
# b.shape --> (1, 3)
# And rule 2 tells us that we upgrade each of these ones to match
# the corresponding size of the other array
# a.shape --> (3, 3)
# b.shape --> (3, 3)
# Because the result matches, these shapes are compatible, we can see this
# here
a + b
(a + b).shape

# Broadcasting example 3
# example in whcih the two arrays are not compatible
M = np.ones((3, 2))
a = np.arange(3)
# This is just a slightly different situation than in the first example:
# The matrix M is transposed. How does this affect the calculation?
M.shape
a.shape

# Again, rule 1 tells us that we must pad the shape of `a` with ones
# M.shape --> (3, 2)
# a.shape --> (1, 3)
# By rule 2, the first dimension of a is stretched to match that of M:
# M.shape --> (3, 2)
# a.shape --> (3, 3)
# Now we hit rule 3, the final shapes do not match, so these two
# arrays are incompatible, as we can observe bu attempting this operation
M + a


# broadcasting applies to any ufunc

#%%
# 
# Broadcasting in Practice
# Centering an array
X = np.random.random((10, 3))
Xmean = X.mean(0)
Xmean
X_centered = X - Xmean
X_centered.mean(0)   # to within-machine precision, the mean is now zero

# plotting a two-dimensional function
# One place that broadcasting is very useful is in displaying images based
# on two.dimensional functions. If we want to define a function z = f(x, y),
# broadcasting can be used to compute the function across the grid
x = np.linspace(0, 5, 500)
y = np.linspace(0, 5, 500)[:, np.newaxis]
z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

%matplotlib
import matplotlib.pyplot as plt

plt.imshow(z, origin = 'lower', extent = [0, 5, 0, 5], cmap = 'viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()


#%%
#
# Comparisons, Masks and Boolean Logic
#
import numpy as np
import pandas as pd

# use pandas to extract rainfall inches as a NumPY array
rainfall = pd.read_csv('data/Seattle2014.csv')['PRCP'].values
inches = rainfall / 254  # 1/10mm -> inches
inches.shape

%matplotlib
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
plt.hist(inches, 40)


# All six of the standard comparison operations are available
# for ufuncs
x = np.array([1, 2, 3, 4, 5])
x < 3
x == 3
x >= 3
x != 3

(2 * x) == (x ** 2)

rng = np.random.RandomState(0)
x = rng.randint(10, size = (3, 4))
x
x < 6


# Counting entries
# count the number of True entries in a Boolean array
np.count_nonzero(x < 6)

# we see that there are eight array entries that are less than 6
# Another way to get at this information is to use np.sum; in this case
# False is interpreted as 0, True as 1
np.sum(x < 6)

# How many values less than 6 in each row?
np.sum(x < 6, axis = 1)

# are there any values greater than 8?
np.any(x > 8)

# ate there any values less than zero?
np.any(x < 0)

# are all values equal to 6?
np.all(x == 6)

# are all values in each row less thgan 8?
np.all(x < 8, axis = 1)

# Boolean operators
np.sum((inches > 0.5) & (inches < 1))

# same result in different manner
np.sum(~( (inches <= 0.5) | (inches >= 1) ))

# !   !   ! 
# Operator | Equivalent ufunc
#    &        np.bitwise_and
#    |        np.bitwise_or
#    ^        np.bitwise_xor
#    ~        np.bitwise_not
# !   !   !

# Boolean Arrays as Masks
# Construct a mask of all rainy days
rainy = (inches > 0)

# construct a mask of all summer days (June 21st is the 172nd day)
summer = (np.arange(365) - 172 < 90) & (np.arange(365) - 172 > 0)
print("Median precip on rainy days in 2014 (inches):", np.median(inches[rainy]))
print("Median precip on summer days in 2014 (inches): ", np.median(inches[summer]))
print("Maximum precip on summer days in 2014 (inches): ", np.max(inches[summer]))
print("Median precip on non-summer rainy days (inches):", np.median(inches[rainy & ~summer]))


#%%
# Difference between & - and , | - or
# So remember this: and and or perform a single Boolean evaluation on an entire
# object, while & and | perform multiple Boolean evaluations on the content (the indi‐
# vidual bits or bytes) of an object. For Boolean NumPy arrays, the latter is nearly
# always the desired operation.

# %%
# Fancy Indexing
#
# Fancy indexing is conceptually simple: it means an array of indices
# to access multiple array elements at once
import numpy as np
rand = np.random.RandomState(42)
x = rand.randint(100, size = 10)
print(x)

ind = [3, 7, 4]
x[ind]

# With fancy indexing, the shape of the result reflefcts the shape of the index
# arrays rather than the shape of the array being indexed
ind = np.array([[3, 7],
               [3, 5]])
x[ind]

# Fancy idnexing also works in multiple dimensions
X = np.arange(12).reshape((3, 4))
X

# Like with standard indexing, the first index refers to the row, and the
# second to the column
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
X[row, col]

# if we combine a column vector and a row vector within the indices, we
# get a two-dimensional result:
X[row[:, np.newaxis], col]

row[:, np.newaxis] * col

#%%
# Combined indexing
print(X)
X[2, [2, 0, 1]]
X[1:, [2, 0, 1]]

mask = np.array([1, 0, 1, 0], dtype = bool)
X[row[:, np.newaxis], mask]

# Example: Selecting Random Points
mean = [0, 0]
cov = [[1, 2],
       [2, 5]]
X = rand.multivariate_normal(mean, cov, 100)
X.shape

import matplotlib.pyplot as plt
import seaborn
seaborn.set()

plt.scatter(X[:, 0], X[:, 1])

# lets choose 20 random points
indices = np.random.choice(X.shape[0], 20, replace = False)
indices
selection = X[indices]    # fancy idnexing here
selection.shape

# visualzie selected rabdom points
plt.scatter(X[:, 0], X[:, 1], alpha = 0.3)
plt.scatter(selection[:, 0], selection[:, 1], facecolor = 'none', edgecolor = 'red', s = 80)

# %%
#
# Modifying values with Fancy Indexing
x = np.arange(10)
i = np.array([2, 1, 8, 4])
x[i] = 99
print(x)
x[i] -= 10
print(x)

# notice, though, that repeated indices with these operations can cause
# some potentially unexpected results, consider the following
x = np.zeros(10)
x[[0, 0]] = [4, 6]
print(x)
# where did the 4 go? The result of this operation is to first assign x[0] = 4,
# followed by x[0] = 6, the result, of course, is that x[0] containts the value 6

# fait enough, but cconsider this operation
i = [2, 3, 3, 4, 4, 4, 5]
x[i] += 1
print(x)

# You might expect that x[3] would contain the value 2, and x[4] would contain the
# value 3, as this is how many times each index is repeated. Why is this not the case?
# Conceptually, this is because x[i] += 1 is meant as a shorthand of x[i] = x[i] + 1 .
# x[i] + 1 is evaluated, and then the result is assigned to the indices in x . With this in
# mind, it is not the augmentation that happens multiple times, but the assignment,
# which leads to the rather nonintuitive results.

# So what if you want the other behavior where the operation is repeated? For this, you
# can use the at() method of ufuncs (available since NumPy 1.8), and do the following:
x = np.zeros(10)
np.add.at(x, i, 1)
print(x)

#%%
#
# Example:
# Binning data
np.random.seed(42)
x = np.random.randn(100)

# compute a histogram by hand
bins = np.linspace(-5, 5, 20)
counts = np.zeros_like(bins)

# find the appropriate bin for each x
i = np.searchsorted(bins, x)

# add 1 to each of these bins
np.add.at(counts, i, 1)

# plot the results
plt.plot(bins, counts, linestyle = 'steps')

plt.hist(x, bins, histtype='step');


#
# %%
# 
# SORTING ARRAYS
#
# Selection Sort:
# Selection sort repeatedly finds the minimum value from a lsit,
# and makes swaps until the list is sorted
import numpy as np
def selection_sort(x):
    for i in range(len(x)):
        # print('i', i)
        # print('x[i:]', x[i:])
        # print('np.argmin(x[i:])', np.argmin(x[i:]))
        swap = i + np.argmin(x[i:])
        # print('swap', swap)
        (x[i], x[swap]) = (x[swap], x[i])
    return x

x = np.array(np.random.randint(-100, 100, 10))
print(bogosort(x))

# bogosort
def bogosort(x):
    while np.any(x[:-1] > x[1:]):
        np.random.shuffle(x)
    return x
print(bogosort(x))

# numpy implementations
x = np.array([2, 1, 4, 3, 5])
np.sort(x)    # sort array without modifying input
x.sort()      # sort in place
x

# a related function is argsort, whichinstead returns the indices
# of the sorted elements
x = np.array([2, 1, 4, 3, 5])
i = np.argsort(x)
print(i)
x[i]

#%%
# Sorting along rows or columns
import numpy as np
rand = np.random.RandomState(42)
X = rand.randint(0, 10, (3, 6))
print(X)

# sort each column of X
np.sort(X, axis = 0)

# sort each row of X
np.sort(X, axis = 1)

#%%
# Partial Sorts: Partitioning
#
# np.partition takes an array and a number K, the result is a new
# array with the smallest K values to the left of the partition
x = np.array([7, 2, 3, 1, 6, 5, 4])
np.partition(x, 3)
np.partition(X, 2, axis = 1)

# there also exist an np.argpartition that computes indices of the partiton

#%%
# K nearest Neighbors
import matplotlib.pyplot as plt
import seaborn
%matplotlib
seaborn.set()
X = rand.rand(100, 2)
# plt.scatter(X[:, 0], X[:, 1], s = 100)

# compute distance for each pair of points
dist_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis = -1)

# the line above can be brken into
# for each pair of ponts, compute differences in their coordinates
differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
differences.shape

# square the coordinate differences
sq_differences = differences ** 2
sq_differences.shape

# sum the coordinate differences to het the squared distance
dist_sq = sq_differences.sum(-1)
dist_sq.shape

# just to double-check what we are doing, we should see that the diagonal of this
# matrix (the set of distances between each point and itself) is all zero
dist_sq.diagonal()

# get nearest neighbors
nearest = np.argsort(dist_sq, axis = 1)

# by using a full sort here, we've actually done more work than we need to in
# this case. If we're simple interested in the nearest k neighbors, all we need
# is to partition each row so that the smalled k + 1 squared distances come first
# with larget distances filling the reamining positions of the array
K = 2
nearest_partition = np.argpartition(dist_sq, K + 1, axis =  1)
print(nearest_partition)
nearest_sort = np.sort(dist_sq, axis = 1)

# plot results
plt.scatter(X[:, 0], X[:, 1], s = 100)

# draw lines from each point to its two nearest neighbors
for i in range(X.shape[0]):
    for j in nearest_partition[i, :K + 1]:
        # plot a line from x[i] to x[j]
        # use some zip magic to make it happen
        plt.plot(*zip(X[j], X[i]), color = 'k')
        
        
    
#%%
# structured arrays in numpy
#
data = np.zeros(4, dtype = {'names': ('name', 'age', 'weight'),
                            'formats': ('U10', 'i4', 'f8')})
print(data.dtype)
# U10 --> Unicode string of maximum length 10
# i4  --> 4-byte (32 bit) integer
# f8  --> 8-byte (64 bit) float

# we can fill the array
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)

# get all names
data['name']
data[0]
data[-1]['name']

# get names where age is under 30
data[data['age'] < 30]['name']

#%%
# More advanced Compount Types
tp = np.dtype([('id', 'i8'), ('mat', 'f8', (3, 3))])
X = np.zeros(1, dtype = tp)
print(X[0])
print(X['mat'][0])