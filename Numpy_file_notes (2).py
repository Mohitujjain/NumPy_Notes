#!/usr/bin/env python
# coding: utf-8

# ###  """What is NumPy?
# NumPy stands for numeric python which is a python package for the computation and processing of the
# multidimensional and single dimensional array elements.
# Travis Oliphant created NumPy package in 2005 by injecting the features of the ancestor module Numeric
# into another module Numarray.
# NumPy is a general-purpose array-processing package .It provides a high-performance multidimensional array
# object ,and tools for working with these arrays.It is the fundamental package for scientific computing
# with python.
# 
# #####
# NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.
# 
# At the core of the NumPy package, is the ndarray object. This encapsulates n-dimensional arrays of homogeneous data types, with many operations being performed in compiled code for performance.
# 
# 
# NumPy gives us the best of both worlds: element-by-element operations are the “default mode” when an ndarray is involved, but the element-by-element operation is speedily executed by pre-compiled C code. In NumPy.
# 
# 
# ### Why is NumPy Fast?
# 
# Vectorization describes the absence of any explicit looping, indexing, etc., in the code - these things are taking place, of course, just “behind the scenes” in optimized, pre-compiled C code. Vectorized code has many advantages, among which are:
# 
# * vectorized code is more concise and easier to read
# 
# * fewer lines of code generally means fewer bugs
# 
# * the code more closely resembles standard mathematical notation (making it easier, typically, to     correctly code mathematical constructs)
# 
# * vectorization results in more “Pythonic” code. Without vectorization, our code would be littered with inefficient and difficult to read for loops.
# 
# Broadcasting is the term used to describe the implicit element-by-element behavior of operations; generally speaking, in NumPy all operations, not just arithmetic operations, but logical, bit-wise, functional, etc., behave in this implicit element-by-element fashion, i.e., they broadcast. Moreover, in the example above, a and b could be multidimensional arrays of the same shape, or a scalar and an array, or even two arrays of with different shapes, provided that the smaller array is “expandable” to the shape of the larger in such a way that the resulting broadcast is unambiguous. For detailed “rules” of broadcasting see Broadcasting.
# #####
# 
# NumPy module works with numerical data.
# NumPy has a powerful tool like Arrays.
# NumPy is used in the popular organizations like SweepSouth.
# NumPy has a better performance for 50k rows or less.
# Numpy consumes less memory as compared to pandas.
# Numpy is mentioned in 62company stacks and 32 developer stacks.
# Numpy provides a multi-dimensional array.
# 
# ### What is an array?
# 
# An array is a data structure that stores values of same data type.In Python, this is the main
# difference betweeen arrays and lists.While python lists can contain values corresponding to different data types arrays in python can only contain values corresponding to same data type.
# 
# 
# 
# ###  Why use NumPy?
# 
# NumPy arrays are faster and more compact than Python lists. An array consumes less memory and is convenient to use.
# NumPy uses much less memory to store data and it provides a mechanism of specifying the data types. 
# This allows the code to be optimized even further.To work with ndarrays, we need to load the numpy library.
# It is standard practice to load numpy with the alias "np" like so:
# """
# 
# 
# ### Who Else Uses NumPy?
# 
# NumPy fully supports an object-oriented approach, starting, once again, with ndarray. For example, ndarray is a class, possessing numerous methods and attributes. Many of its methods are mirrored by functions in the outer-most NumPy namespace, allowing the programmer to code in whichever paradigm they prefer. This flexibility has allowed the NumPy array dialect and NumPy ndarray class to become the de-facto language of multi-dimensional data interchange used in Python.
# 
# ### SEED
# The numpy random seed is a numerical value that generates a new set or repeats pseudo-random numbers. The value in the numpy random seed saves the state of randomness. If we call the seed function using value 1 multiple times, the computer displays the same random numbers.

# In[1]:


#################################################################################################


# 
# ###### NumPy is a general-purpose array-processing package. It provides a high-performance multidimensional array object, and tools for working with these arrays. It is the fundamental package for scientific computing with Python

# ###### What is an array
# An array is a data structure that stores values of same data type. In Python, this is the main difference between arrays and lists. While python lists can contain values corresponding to different data types, arrays in python can only contain values corresponding to same data type

# In[2]:


## import the library
import numpy as np


# In[3]:


lst=[1,2,3,4]
arr=np.array(lst)


# In[4]:


type(arr)


# In[5]:


arr.shape


# In[6]:


lst1=[1,2,3,4,5]
lst2=[2,3,4,5,6]
lst3=[3,4,5,6,7]

arr1=np.array([lst1,lst2,lst3])


# In[7]:


arr1


# In[8]:


arr1.shape


# In[9]:


arr


# In[10]:


#indexing
arr[3]


# In[11]:


arr[3]=5


# In[12]:


arr


# In[13]:


arr[-1]


# In[14]:


arr[:-1]


# In[15]:


arr[::-3]


# In[16]:


arr1


# In[17]:


arr1[:,3:].shape


# In[18]:


arr1[:,1]


# In[19]:


arr1[1:,1:3]


# In[20]:


arr1[1:,3:]


# In[21]:


##EDA
arr


# In[22]:


arr[arr<2]


# In[23]:


arr1


# In[24]:


arr1.reshape(5,3)


# In[25]:


##mechanism to create an array
np.arange(1,20,2).reshape(2,5)


# In[26]:


np.arange(1,20,2).reshape(2,5,1)


# In[27]:


arr *arr


# In[28]:


arr1 * arr1


# In[29]:


np.ones((5,3))


# In[30]:


np.zeros((4,5))


# In[31]:


np.random.randint(10,50,4).reshape(2,2)


# In[32]:


np.random.randn(5,6)


# In[33]:


np.random.random_sample((4,7))


# In[34]:


#####################################################################################################


# In[35]:


my_list = [1, 2, 3, 4] # Define a list
my_array = np.array(my_list) # Pass the list to np.array()
type(my_array) # Check the object's type


# In[36]:


'''
shape : gives row by columns
ndim : rank of the array
The number of dimensions is the rank of the array; the shape of an array is a
tuple of integers giving the size of the array along each dimension.
'''
array1D = np.array( [1,2,3,4] )
print('1D array \n', array1D) # 1D array | vector
print('Shape : ', array1D.shape) # (4,)
print('Rank : ', array1D.ndim) # 1
print('Size : ', array1D.size) # 4
print('DatA Type : ', array1D.dtype) # int
print('--------------------------------------------')
array2D = np.array( [ [1.,2.,3.,4.], [4.,3.,2.,1.] ] )
print('2D array \n',array2D) # 2D array | matrix
print('Shape : ', array2D.shape) # (2,4)
print('Rank : ', array2D.ndim) # 2
print('Size : ', array2D.size) # 8
print('DatA Type : ', array2D.dtype) # float
print('--------------------------------------------')
array3D = np.array( [ [[1,2,3,4]], [[-1,-2,-3,-4]], [[1,2,3,4]] ] )
print('3D array \n', array3D)
print('Shape : ', array3D.shape) # (3, 1, 4)
print('Rank : ', array3D.ndim) # 3
print('Size : ', array3D.size) # 12
print('DatA Type : ', array3D.dtype) # int


# In[37]:


'''
arange : Return evenly spaced values within a given interval. (doc words)
reshape : Gives a new shape to an array without changing its data. (doc words)
'''
numbers = np.arange(10) # It will create a 1D numpy array
print(numbers) # 0,1,2,3,4,5,6,7,8,9
print(numbers.dtype) # int
print(type(numbers)) # numpy.ndarray
print('--------------------------------------------')
reshape_number = numbers.reshape(2,5) # It'll create a 2D numpy array.
print(reshape_number) # [[0 1 2 3 4] [5 6 7 8 9]]
print(reshape_number.dtype) # int
print(type(reshape_number)) # numpy.ndarray
print('--------------------------------------------')
array2D = np.arange(20).reshape(4,5) # Create 2D array with shape (4,5) from 0 to 19
print('2D array \n',array2D) # 2D array | matrix
print('Shape : ', array2D.shape) # (4,5)
print('Rank : ', array2D.ndim) # 2
print('Size : ', array2D.size) # 20
print('DatA Type : ', array2D.dtype) # int
print('--------------------------------------------')
array3D = np.arange(20).reshape(2, 2, 5) # Create 3D array with shape (2,2,5) from 0 to 19
print('3D array \n',array3D) # 2D array | matrix
print('Shape : ', array3D.shape) # (2, 2, 5) | (channel , width, height) ; we've two 2 by 5 matrix
print('Rank : ', array3D.ndim) # 3
print('Size : ', array3D.size) # 20
print('DatA Type : ', array3D.dtype) # int


# In[38]:


all_zeros = np.zeros((2,2)) # Create an array of all zeros
print('All Zeros \n' ,all_zeros) # Prints "[[ 0. 0.]
 # [ 0. 0.]]"
print('--------------------------------------------')

all_ones = np.ones((1,2)) # Create an array of all ones
print('All Ones \n', all_ones) # Prints "[[ 1. 1.]]"
print('--------------------------------------------')
filled_array = np.full((2,2), 3) # Create a constant array
print('Filled with specified valued \n', filled_array) # Prints "[[ 3. 3.]
 # [ 3. 3.]]"
print('--------------------------------------------')

identity_mat = np.eye(2) # Create a 2x2 identity matrix
print('Identity Matrix \n', identity_mat) # Prints "[[ 1. 0.]
 # [ 0. 1.]]"
print('--------------------------------------------')

random_normal_distro = np.random.random((2,2)) # Create an array filled with random values
print('Normal Distribution \n', random_normal_distro)
print('--------------------------------------------')
evenly_spaced_ranged_number = np.linspace(1,3,10) # range 1 to 3, generate 10 digit with evely spaced
print('Evely spaced number in givend range \n', evenly_spaced_ranged_number)
print('--------------------------------------------')
linspace_reshape = np.linspace(1,3,10).reshape(2,5)
print('2D array \n',linspace_reshape) # 2D array | matrix
print('Shape : ', linspace_reshape.shape) # (2, 5)
print('Rank : ', linspace_reshape.ndim) # 2
print('Size : ', linspace_reshape.size) # 10
print('DatA Type : ', linspace_reshape.dtype) # float
print('Converted Data Type : ', linspace_reshape.astype('int64').dtype) # convert float to int
print('2D array \n',linspace_reshape.astype('int64')) # But this will truncated numbers after decimal


# In[39]:


"""What is an array?
An array is a central data structure of the NumPy library. An array is a grid of values and
it contains information about the raw data, how to locate an element, and how to interpret an element.
It has a grid of elements that can be indexed in various ways.

An array can be indexed by a tuple of nonnegative integers, by booleans, by another array, or by integers.
The rank of the array is the number of dimensions. The shape of the array is a tuple of integers giving
the size of the array along each dimension.

OR

1. Arrays in NumPy: NumPy’s main object is the homogeneous multidimensional array.
2. It is a table of elements (usually numbers), all of the same type, indexed by a tuple of positive 
   integers.
3. In NumPy dimensions are called axes. The number of axes is rank.
4. NumPy’s array class is called ndarray. It is also known by the alias array.
  One way we can initialize NumPy arrays is from Python lists, using nested lists for two- or
  higher-dimensional data.Numpy offers several ways to index into arrays. We may want to select a subset of our data or individual elements. Most common ways are:
  Slicing
 Integer Array Indexing / Fnacy Indexing
 Boolean Indexing
 Slicing Like in Python lists, NumPy arrays can be sliced.



"""


# In[40]:


array2D = np.arange(0,40,2).reshape(4,5)
print(array2D) # shape : (4,5)


# In[41]:


'''
Use slicing to pull out the subarray from the orignial array.
Le's say we want to get following sub-array from array2D.
This located at row (1,2) and column (1,2).


[12 14]
[22 24]
So, we need to do somthing like array2D[row-range, column-range]. Note that,
while indexing we need to range 1 step more, as we do
in np.arange(0,10) <- go 0 to 9 but not 10.
'''
# and columns 1 and 2; b is the following array of shape (2, 2):
sliced_array_1 = array2D[1:3, 1:3] # look we set 1:3 <- go 1 to 2 but not include 3
print(sliced_array_1)
print('--------------------------------------------')
sliced_array_2 = array2D[:, 1:4] # The 'bare' slice [:] will asign to all values in an array
print(sliced_array_2)
print('--------------------------------------------')
sliced_array_3 = array2D[:4, 2:] # row: 0 to 3 ; column: 2 to all
print(sliced_array_3)
'''
More practice. array2D:
[[ 0 2 4 6 8]
 [10 12 14 16 18]
 [20 22 24 26 28]
 [30 32 34 36 38]]
Let's get some specific portion.
1. [16 18],
 [26 28]
    
2. [20 22 24],
 [30 32 34]

3. [14 16],
 [24 26],
  [34 36]
'''
print('--------------------------------------------')
sliced_array_4 = array2D[1:3, 3:] # row: 1 to 2 ; column: 3 to all
print('1 \n', sliced_array_4)
print('--------------------------------------------')
sliced_array_5 = array2D[2:, 0:3] # row: 2 to all ; column: 0 to 2
print('2 \n', sliced_array_5)
print('--------------------------------------------')
sliced_array_6 = array2D[1:, 2:4] # row: 1 to all ; column: 2 to 3
print('3 \n', sliced_array_6)


# In[42]:


# Python program to demonstrate
# basic array characteristics
# Creating array object
arr = np.array( [[ 1, 2, 3],
 [ 4, 2, 5]] )
# Printing type of arr object
print("Array is of type: ", type(arr))
# Printing array dimensions (axes)
print("No. of dimensions: ", arr.ndim)
# Printing shape of array
print("Shape of array: ", arr.shape)
# Printing size (total number of elements) of array
print("Size of array: ", arr.size)
# Printing type of elements in array
print("Array stores elements of type: ", arr.dtype)


# In[43]:


import numpy as np
a = np.array([1,2,3,4])
 #OR
b = np.array([[1,2,3,4],[5,6,7,7],[8,9,10,11]])
print(b[0])


# In[44]:


### We can access the elements in the array using square brackets. When you’re accessing elements,
 ## remember that indexing in NumPy starts at 0.
## That means that if you want to access the first element in your array, you’ll be accessing element “0”.


# In[45]:


""" 
“ndarray,” which is shorthand for “N-dimensional array.” An N-dimensional array is simply an array with
 any number of dimensions. You might also hear 1-D, or one-dimensional array, 2-D, or two-dimensional array, and so on. 
The NumPy ndarray class is used to represent both matrices and vectors. A vector is an array with a single
dimension (there’s no difference between row and column vectors), while a matrix refers to an array with 
two dimensions. For 3-D or higher dimensional arrays, the term tensor is also commonly used.
"""


# In[46]:


get_ipython().set_next_input('How To Make An “Empty” NumPy Array');get_ipython().run_line_magic('pinfo', 'Array')


# In[ ]:


How To Make An “Empty” NumPy Array


# In[47]:


How To Make An “Empty” NumPy Array


# In[50]:


# Create an array of ones
np.ones((3,4))
# Create an array of zeros
np.zeros((2,3,4),dtype=np.int16)
# Create an array with random values
np.random.random((2,2))
# Create an empty array
np.empty((3,2))
# Create a full array
np.full((2,2),7)
# Create an array of evenly-spaced values
np.arange(10,25,5)

# Create an array of evenly-spaced values
np.linspace(0,2,9)


# In[ ]:


## Array creation: There are various ways to create arrays in NumPy.
 #For example, you can create an array from a regular Python list or tuple using the array function.
 #   The type of the resulting array is deduced from the type of the elements in the sequences.


# In[ ]:


# Often, the elements of an array are originally unknown, but its size is known. Hence, NumPy offers 
# several functions to create arrays with initial placeholder content. These minimize the necessity of
# growing arrays, an expensive operation.
# For example: np.zeros, np.ones, np.full, np.empty, etc.
# To create sequences of numbers, NumPy provides a function analogous to range that returns arrays instead
  # of lists.
 # arange: returns evenly spaced values within a given interval. step size is specified.
# linspace: returns evenly spaced values within a given interval. num no. of elements are returned.
# Reshaping array: We can use reshape method to reshape an array. Consider an array with shape (a1, a2, a3, …, aN). We can reshape and
# convert it into another array with shape (b1, b2, b3, …, bM). The only required condition is:
# a1 x a2 x a3 … x aN = b1 x b2 x b3 … x bM . (i.e original size of array remains unchanged.)
# Flatten array: We can use flatten method to get a copy of array collapsed into one dimension. It accepts order argument. Default value is ‘C’
# (for row-major order). Use ‘F’ for column major order.
# Note: Type of array can be explicitly defined while creating array


# In[ ]:


# Python program to demonstrate
# array creation techniques
import numpy as np
# Creating array from list with type float 
a = np.array([[1, 2, 4], [5, 8, 7]], dtype= 'float')
print ("Array created using passed list:\n",a)
# Creating array from tuple 
b = np.array((1 , 3, 2))
print ("\nArray created using passed tuple:\n",b)
# Creating a 3X4 array with all zeros
c = np.zeros((3, 4))
print ("\nAn array initialized with all zeros:\n",c)
# Create a constant value array of complex type d = np.full((3, 3), 6, dtype = 'complex')
print ("\nAn array initialized with all 6s."
           "Array type is complex:\n",d)
# Create an array with random values 
e = np.random.random((2, 2))
print ("\nA random array:\n",e)
# Create a sequence of integers
# from 0 to 30 with steps of 5 
f = np.arange(0, 30, 5)
print ("\nA sequential array with steps of 5:\n", f )
# Create a sequence of 10 values in range 0 to 5
g = np.linspace(0, 5, 10)
print ("\nA sequential array with 10 values between"
                               "0 and 5:\n", g )
# Reshaping 3X4 array to 2X2X3 array
arr= np.array([[1,2,3,4],
               [5,2,4,2],
               [1,2,0,1]])
newarr= arr.reshape(2,2,3)


print ("\nOriginal array:\n", arr)
print ("Reshaped array:\n", newarr)
# Flatten array
arr = np.array([[1, 2, 3], [4, 5, 6]])
flarr = arr.flatten()
print ("\nOriginal array:\n", arr)
print ("Fattened array:\n", flarr)


# In[ ]:


# python program for creation of array
import numpy as np
# creating Rank 1 array
arr = np.array([1,2,3])
print("array with rank 1 : \n",arr)
# Creating a rank 2 Array
arr = np.array([[1,2,3],
 [4,5,6]])
print("arrray with rank 2 : \n",arr)
## Creating an array from tuple
arr = np.array((1, 3, 2))
print("\nArray created using "
 "passed tuple:\n", arr)


# In[ ]:


a = np.array([2,4,6])
print(a)


# In[ ]:


a = np.array([2,4,6])
print(a)
value = a[2]
print(value)


# In[ ]:


### Multi-dimensional Array Indexing
 # Multi-dimensional arrays can be indexed as well. A simple 2-D array is defined by a list of lists.
a = np.array([[2,3,4],[6,7,8]])
print(a)


# In[ ]:


a = np.array([[2,3,4],[6,7,8]])
print(a)
value = a[1,2]
print(value)


# In[ ]:


a = np.array([2,4,6])
a[2] = 10
print(a)


# In[ ]:


a = np.array([[2,3,4],[6,7,8]])
print(a)
print("***************")
a[1,2] = 20
print(a)


# In[ ]:


a = np.array([2,4,6])
b = a[0:2]
print(b)


# In[ ]:


a = np.array([2, 4, 6, 8])
print(a)
b = a[1:]
print(b)


# In[51]:


a = np.array([2, 4, 6, 8])
print(a)
b = a[:3]
print(b)


# In[52]:


a = np.array([2, 4, 6, 8])
b = a[0:4]
print(b)
c = a[:4]
print(c)
d = a[0:]
print(d)
e = a[:]
print(e)


# In[53]:


a = np.array([[2, 4, 6, 8], [10, 20, 30, 40]])
print(a)
b = a[0:2, 0:3]
print(b)


# In[ ]:


a = np.array([[2, 4, 6, 8], [10, 20, 30, 40]])
b = a[:2, :] #[first two rows, all columns]
print(b)


# In[ ]:


bool_index = np.arange(32).reshape(4,8)
print(bool_index)


# In[ ]:


bool_index < 20 # boolean expression


# In[54]:


'''
The often we use such operaton when we do thresholding on the data. We can use these
operation as an index and can get the result based on the expression. Let's see some
of the examples.
'''
bool_index[ bool_index < 20 ] # only get the values which is less than 20


# In[ ]:


'''
let's see some various example of using this.
'''
print(bool_index[ bool_index % 2 == 0 ]) # [ 0 2 4 ... 28 30]
print(bool_index[ bool_index % 2 != 0 ]) # [ 1 3 5 ... 29 31]
print(bool_index[ bool_index % 2 == 0 ] + 1) # [ 1 3 5 ... 29 31]


# In[ ]:


x = np.arange(0,4).reshape(2,2).astype('float64')
y = np.arange(5,9).reshape(2,2).astype('float64')
# Elementwise sum; both produce the array
print(x + y)
print(np.add(x, y))
print('----------------------------')
# Elementwise difference; both produce the array
print(x - y)
print(np.subtract(x, y))
print('----------------------------')
# Elementwise product; both produce the array
print(x * y)
print(np.multiply(x, y))
print('----------------------------')
# Elementwise division; both produce the array
print(x / y)
print(np.divide(x, y))
print('----------------------------')
# Elementwise square root; produces the array
print(np.sqrt(x))


# In[55]:


'''
Numpy provides many useful functions for performing computations on arrays;
one of the most useful is sum
'''
x = np.arange(5,9).reshape(2,2).astype('int64')
print(x)
print(np.sum(x)) # Compute sum of all elements; prints "26"
print(np.sum(x, axis=0)) # Compute sum of each column; prints "[12 14]"
print(np.sum(x, axis=1)) # Compute sum of each row; prints "[11 15]"


# In[ ]:


'''
Sometimes we need to manipulate the data in array. It can be done by reshaping or transpose
the array. Transposing is a special form of reshaping that similarly returns a view on the
underlying data without copying anything.
When doing matrix computation, we may do this very often.
'''
arr = np.arange(10).reshape(2,5)
print('At first \n', arr) # At first
 # [[0 1 2 3 4]
# [5 6 7 8 9]]
print()
print('After transpose \n', arr.T)
print('----------------------------')
transpose = np.arange(10).reshape(2,5)
print(np.dot(transpose.T, transpose)) 


# In[ ]:


'''
statistical functions and concern used function, such as
- mean
- min
- sum
- std
- median
- argmin, argmax
'''
ary = 10 * np.random.randn(2,5)
print('Mean : ', np.mean(ary)) # 0.9414738037734729
print('STD : ', np.std(ary)) # 5.897885490589387
print('Median : ', np.median(ary)) # 1.5337461352996276
print('Argmin : ', np.argmin(ary)) # 3
print('Argmax : ', np.argmax(ary)) # 2
print('Max : ', np.max(ary)) # 10.399663734487659
print('Min : ', np.min(ary)) # -9.849839643044087
print('Compute mean by column :', np.mean(ary, axis = 0)) # compute the means by column
print('Compute median by row : ', np.median(ary, axis = 1)) # compute the medians


# In[ ]:


ary = np.arange(5)
print('Find root of each elements-wise \n', np.sqrt(ary))
print()
print('Find exponential for each element-wise \n', np.exp(ary))
# Find root of each elements-wise
# [0. 1. 1.41421356 1.73205081 2. ]
# Find exponential for each element-wise
# [ 1. 2.71828183 7.3890561 20.08553692 54.59815003]
'''
np.maximum
This computed the element-wise maximum of the elements in two array and returned a single
array as a result.
'''
print()
print('Max values between two array \n', np.maximum(np.sqrt(ary), np.exp(ary)))
print('----------------------------')
print()
# Max values between two array
# [ 1. 2.71828183 7.3890561 20.08553692 54.59815003]
'''
np.modf
Another unfunc but can return multiple arrays. It returns the fractional and integral parts
of a floating point array.
'''
rem, num = np.modf(np.exp(ary))
print('Floating Number ', np.exp(ary))
print()
print('Remainder ', rem)
print('Number ', num)
print('----------------------------')
print()


# In[ ]:


# Floating Number [ 1. 2.71828183 7.3890561 20.08553692 54.59815003]
# Remainder [0. 0.71828183 0.3890561 0.08553692 0.59815003]
# Number [ 1. 2. 7. 20. 54.]
'''
np.ceil
Return the ceiling of the input, element-wise.
'''
ceil_num = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
print(ceil_num) # [-1.7 -1.5 -0.2 0.2 1.5 1.7 2. ]
print(np.ceil(ceil_num)) # [-1. -1. -0. 1. 2. 2. 2.]
''' not ufunc
np.around
Evenly round to the given number of decimals.
'''
print(np.around(ceil_num)) # [-2. -2. -0. 0. 2. 2. 2.]
print('----------------------------')
print()
'''
np.absolute | np.abs | np.fabs
Calculate the absolute value element-wise.
'''
absl = np.array([-1, 2])
print('Absolute of Real Values ', np.abs(absl))
print('Absolute of Real Values with Float ', np.fabs(absl))
print('Absolute of Complex Values ', np.abs(1.2 + 1j))
# Absolute of Real Values [1 2]
# Absolute of Real Values with Float [1. 2.]
# Absolute of Complex Values 1.5620499351813308


# In[ ]:


'''
np.random.rand()
Create an array of the given shape and populate it with
random samples from a uniform distribution
over ``[0, 1)``
'''
ary = np.random.rand(5,2) # shape: 5 row, 2 column
print('np.random.rand() \n', ary)
print('----------------------------')
print()
'''
np.random.randn
Return a sample (or samples) from the "standard normal" distribution.
'''
ary = np.random.randn(6)
print('1D array: np.random.randn() \n', ary)
ary = np.random.randn(3,3)
print('2D array: np.random.randn() \n', ary)
print('----------------------------')
print()
'''
np.random.random.
numpy.random.random() is actually an alias for numpy.random.random_sample()
Return a sample (or samples) from the "standard normal" distribution.
'''
ary = np.random.random((3,3))
print('np.random.randn() \n', ary)
ary = np.random.random_sample((3,3))
print('np.random.random_sample() \n', ary)
print('----------------------------')
print()
'''
np.random.randint.
Return random integers from low (inclusive) to high (exclusive)
Return random integers from the “discrete uniform” distribution of the specified
dtype in the “half-open” interval [low, high). If high is None (the default),
then results are from [0, low).
'''
ary = np.random.randint(low = 2, high = 6, size = (5,5))
print('np.random.randint() \n', ary)
ary = np.random.randint(low = 2, high = 6)
print('np.random.randint() :', ary)


# In[ ]:


'''
np.random.normal()
Draw random samples from a normal (Gaussian) distribution. This is Distribution is
also known as Bell Curve because of its characteristics shape.
'''
mu, sigma = 0, 0.1 # mean and standard deviation
print('np.random.normal() \n',np.random.normal(mu, sigma, 10)) # from doc
print('----------------------------')
print()
'''
np.random.uniform()
Draw samples from a uniform distribution
'''
print('np.random.uniform() \n', np.random.uniform(-1,0,10))
print('----------------------------')
print()
'''
np.random.seed()
'''
np.random.seed(3) # seed the result
'''
np.random.shuffle
Modify a sequence in-place by shuffling its contents
'''
ary = np.arange(9).reshape((3, 3))
print('Before Shuffling \n', ary)
np.random.shuffle(ary)
print('After Shuffling \n', ary)
print('----------------------------')
print()
'''
np.random.choice
Generates a random sample from a given 1-D array
'''
ary = np.random.choice(5, 3) # Generate a uniform random sample from np.arange(5) of size 3:
print('np.random.choice() \n', ary) # This is equivalent to np.random.randint(0,5,3)


# In[ ]:


'''
sort()
'''
# create a 10 element array of randoms
unsorted = np.random.randn(10)
print('Unsorted \n', unsorted)
# inplace sorting
unsorted.sort()
print('Sorted \n', unsorted)
print()
print('----------------------------------')
'''
unique()
'''
ary = np.array([1,2,1,4,2,1,4,2])
print('Unique values : ', np.unique(ary))
print()
print('----------------------------------')
'''
vstack and hstack
'''
arx = np.array([[1,2,3],[3,4,5]])
ary = np.array([[4,5,6],[7,8,9]])
print('Vertical Stack \n', np.vstack((arx,ary)))
print('Horizontal Stack \n', np.hstack((arx,ary)))
print('Concate along columns \n', np.concatenate([arx, ary], axis = 0)) # similar vstack
print('Concate along rows \n', np.concatenate([arx, ary], axis = 1)) # similar hstack
print()
print('----------------------------------')
'''
ravel : convert one numpy array into a single column
'''
ary = np.array([[1,2,3],[3,4,5]])
print('Ravel \n', ary.ravel())
print()
print('----------------------------------')
'''
tile()
'''
ary = np.array([-1, 0, 1])
ary_tile = np.tile(ary, (4, 1)) # Stack 4 copies of v on top of each other
print('tile array \n', ary_tile)


# In[ ]:


# Set Function
s1 = np.array(['desk','chair','bulb'])
s2 = np.array(['lamp','bulb','chair'])
print(s1, s2)
print( np.intersect1d(s1, s2) )
print( np.union1d(s1, s2) )
print( np.setdiff1d(s1, s2) )# elements in s1 that are not in s2
print( np.in1d(s1, s2) )


# In[ ]:


start = np.zeros((4,3))
print(start)
print('----------------------------------')
# create a rank 1 ndarray with 3 values
add_rows = np.array([1, 0, 2])
y = start + add_rows # add to each row of 'start' using broadcasting
print(y)
print('----------------------------------')
# create an ndarray which is 4 x 1 to broadcast across columns
add_cols = np.array([[0,1,2,3]])
add_cols = add_cols.T
print(add_cols)
print('----------------------------------')
# this will just broadcast in both dimensions
add_scalar = np.array([1])
print(start + add_scalar)


# In[ ]:




