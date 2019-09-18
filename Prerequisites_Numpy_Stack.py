# -*- coding: utf-8 -*-
from __future__ import print_function, division
# Import Library of Numpy
import numpy as np  


# Version of Numpy
np.__version__   

#NumPy is an N-dimensional array type called ndarray
#NumPy rank 1 array
x1 = np.array([1, 2, 3, 4,5])
print ('x1 =', x1) 
print ('------------------------------')
print ('type (x1) =', type (x1))
print ('------------------------------')
print ('x1.shape =', x1.shape)
print ('------------------------------')
print ('x1.size =', x1.size) # number of elements
print ('------------------------------')
print (x1[0],',',x1[1],',',x1[2],',',x1[3]) #Prints "1 2 3 4"
print ('------------------------------')
x1[3] = 0
print (x1) # Change an element
#A data type object (an instance of numpy.dtype class) describes how the bytes in the fixed-size block of memory corresponding to an array item should be interpreted.
'''#int8 = Byte (-128 to 127)
#int16 = Integer (-32768 to 32767)
#int32 = Integer (-2147483648 to +2147483647)
#int64 = Integer (-9223372036854775808 to +9223372036854775807)
#Boolean = (True or False) stored as a byte
#float = Shorthand for float64 : Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
#complex = Shorthand for complex128 : Complex number
#int8, int16, int32, int64 can be replaced by equivalent string 'i1', 'i2','i4', etc. '''
    
x1a = np.array([1, 2, 3, 4] , dtype='float') #Desired data type of array, optional
#x1a = np.array([1, 2, 3, 4] , dtype='i1')
#x1a = np.array([1, 2, 3, 4] , dtype= complex)
print ('x1a =', x1a) 
print ('------------------------------')
print ('type (x1a) =', type (x1a))
print ('------------------------------')
print ('x1a.dtype =', x1a.dtype)
print ('------------------------------')
print ('x1a.itemsize =', x1a.itemsize) # 8 byte
print ('------------------------------')
print ('x1a.size =', x1a.size)
print ('------------------------------')
print ('x1a.shape =', x1a.shape)
print ('------------------------------')
x1b = np.array([[.25, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12],[13, 14, 15, 16]])
print (x1b.dtype) #dtype is float
print ('------------------------------')
print ('x1b.shape =', x1b.shape) #(4,4)
print ('------------------------------')
x1ba = x1b [0:2,1:3]
print ('x1ba [0:2,1:3] : ', x1ba)
print ('------------------------------')
print ('x1b [0,2] : ', x1b [0,2])
print ('------------------------------')
x1b [0,2] = 17
print ('x1b = ', x1b)
print ('------------------------------')
print ('x1ba : ', x1ba) # has changed   3=>17
print ('------------------------------')
x1ba = +x1b [0:2,1:3] #x1ba dosen't depend on x1b
x1b [0,2] = 125
print ('------------------------------')
print ('x1b = ', x1b)
print ('------------------------------')
print ('x1ba : ', x1ba) # dosen't change
print ('------------------------------')
x1c = np.array([[.25, 2, 3, 4],[8],[9, 10, 11, 12],[13, 14, 15, 16]])
print ('x1c.shape =', x1c.shape) #(4,)
print ('------------------------------')
x1d = x1a.copy()
print('x1d =',x1d)
print ('------------------------------')
print ('x1d is x1a =', x1d is x1a) # False
print ('------------------------------')
print('x1d.ndim =',x1d.ndim) # dimension

# More than one dimension
#NumPy rank 2 array
x2 = np.array([[1,2,3,4],[4,3,2,1]])
print (x2) 
print (type (x2))
print (x2.shape)
print (x2[0,0],',',x2[1,3])

#Return a array setting values to zero
x2a = np.zeros((4,2)) 
print ('x2a =',x2a)
x2b = np.zeros_like(x2)
print ('------------------------------')
print('x2b =',x2b)
print ('------------------------------')
x2b1 = np.zeros((2,4,2)) 
print('x2b1 =',x2b1)
print('x2b1.ndim = ',x2b1.ndim)
print ('------------------------------')
#Return a array setting values to one
x2c = np.ones((4,3))
print('x2c =',x2c)
print ('------------------------------')
x2c = np.ones((4,3), dtype='i1')
print('x2c =',x2c)
print ('------------------------------')
x2d = (4,3)
x2e = np.ones(x2d, dtype='int')
print('x2e =',x2e)
print ('------------------------------')

#numpy.full => Return a new array of given shape and type, filled with fill_value.
x2f = np.full((3,3),0)
print('x2f =',x2f)
print ('------------------------------')
x2g = np.full((2,2),1)
print('x2g =',x2g)
print ('------------------------------')
x2h = np.full((7,),6)
print('x2h =',x2h)
print ('------------------------------')
x2i = np.full((5,2),'test')
print('x2i =',x2i)
print ('------------------------------')

#numpy.eye => Return a 2-D array with ones on the diagonal and zeros elsewhere
x2j = np.eye(4)
print('x2j =',x2j)
print ('------------------------------')
x2k = np.eye(2, dtype=int)
print('x2k =',x2k)
print ('------------------------------')
x2l = np.eye(5, k=2, dtype=int)   # k => Index of the diagonal : 0 (the default) refers to the main diagonal
print('x2l =',x2l)
print ('------------------------------')

# array filled with random values
x2m = np.random.random((3,2))
print('x2m =',x2m)
print ('------------------------------')
#Create an array of the given shape and populate it with random samples from a uniform distribution over
x2n = np.random.rand(2,5) # 2,5 are the dimensions of the returned array
print('x2n =',x2n)
print ('------------------------------')
#uniform([low, high, size]) 	Draw samples from a uniform distribution
x2o = np.random.uniform(2,7,(2,2))
print('x2o =',x2o)
print ('------------------------------')
#standard_normal([size]) 	Draw samples from a standard Normal distribution (mean=0, stdev=1)
x2p = np.random.standard_normal((2,2))
print('x2p =',x2p)
print ('------------------------------')

#numpy.arange(start,stop,step, dtype=None)
#Return evenly spaced values within a given interval.
x3 = np.arange(7)
print ('x3 =',x3)
print ('------------------------------')
x3a = np.arange(7.0)
print ('x3a =',x3a)
print ('------------------------------')
x3b = np.arange(4.5)
print ('x3b =',x3b)
print ('------------------------------')
x3c = np.arange(4,19)
print ('x3c =',x3c)
print ('------------------------------')
x3d = np.arange(4,19,3)   # 3 is step
print ('x3d =',x3d)
print ('------------------------------')
x3e = np.arange(4,19,3, dtype=float)
print ('x3e =',x3e)
print ('------------------------------')
x3f = np.arange(4,19,2.4)
print ('x3f =',x3f)
print ('------------------------------')
#numpy.linspace (start, stop, num) => Return evenly spaced numbers over a specified interval
#from numpy import pi
x3g = np.linspace(3.0, 9.0, num=9)
print ('x3g =',x3g)
print ('------------------------------')

# A matrix is a specialized 2-D array that retains its 2-D nature through operations
x3 = np.matrix([[1,2],[3,4]]) # Or x2 = np.matrix('1 2; 3 4')
print(x3)
print (type (x3))
print (x3.shape)
print (x3[1,1], x3[0,1], x3[0,0], x3[1,0])

#Computation on NumPy Arrays
#Dot product of two arrays. Specifically,
#   If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
#   If both a and b are 2-D arrays, it is matrix multiplication, but using matmul or a @ b is preferred.
#  If either a or b is 0-D (scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a * b is preferred.
x4 = np.dot(7,2)
print("x4 = ",x4)
print ('------------------------------')
x5 = np.ones((2,3))
print("x5 = ",x5)
print ('------------------------------')
x6 = np.array([[2,5],[3,7],[9,6]])
x7 = np.dot(x5,x6)
print("x7 = ",x7)
print ('------------------------------')
x8 = x5 @ x6
print("x8 = ",x8)
print ('------------------------------')
x9 = np.array([[2,2],[4,4]])
x10 = np.multiply(x9,x9)
print('x10 = ',x9,'*',x9,' = ',x10)
print ('------------------------------')
print('x9 * x9 = ',x9*x9)
print ('------------------------------')
print('x9 @ x9 = ',x9@x9)
print ('------------------------------')
#numpy.prod => Return the product of array elements over a given axis
x11 = np.prod (x9)
print("x11 = ",x11)
print ('------------------------------')
#The product of an empty array is the neutral element 1
x12 = np.prod([] , dtype='i1')
print("x12 = ",x12)
print ('------------------------------')
# we can also specify the axis over which to multiply
x13 = np.prod([[1, 2, 3],[4, 5, 6],[11, 10, 2]], axis=0)
print("x13 = ",x13)
print ('------------------------------')

#Broadcasting => The term broadcasting describes how numpy treats arrays with different shapes during arithmetic operations
x14 = np.array([1.0, 2.0, 3.0, 4.0])
x15 = x14 + 7
print("x15 = ",x15)
print ('------------------------------')
x16 = np.ones((4,4))
x17 = x14 + x16
print("x17 = ",x17)
print ('------------------------------')
x18 = np.ones((4,1))
x19 = x14 + x18
print("x19 = ",x19)
print ('------------------------------')

#numpy.sum => Sum of array elements over a given axis
x20 = np.sum([1.5, 2.5])
print("x20 = ",x20)
print ('------------------------------')
x21 = np.sum([[2,4],[6,8]])
print("x21 = ",x21)
print ('------------------------------')
x22 = np.sum([[2,4],[6,8]], axis=0)
print("x22 = ",x22)
print ('------------------------------')
x23 = np.sum([[2,4],[6,8]], axis=1)
print("x23 = ",x23)
print ('------------------------------')
#numpy.cumsum => Return the cumulative sum of the elements along a given axis
a = np.array([[1,2,3], [7,8,9]])
x24 = np.cumsum(a)
print("x24 = ",x24)
print ('------------------------------')
x25 = np.cumsum(a, axis=0)  # sum over rows for each of the 3 columns
print("x25 = ",x25)
print ('------------------------------')
x26 = np.cumsum(a, axis=1) # sum over columns for each of the 2 rows
print("x26 = ",x26)
print ('------------------------------')
#numpy.subtract
x27 = np.subtract(a,a)
print("x27 = ",x27)
print ('------------------------------')
#numpy.divide => Returns a true division of the inputs
x28 = np.divide(a,2)
print("x28 = ",x28)
print ('------------------------------')
x29 = np.floor_divide(a,2)
print("x29 = ",x29)
print ('------------------------------')
x30 = np.true_divide(a,2)
print("x30 = ",x30)
print ('------------------------------')

#numpy.math
'''
1- sin(x, /[, out, where, casting, order, ...]) 	Trigonometric sine, element-wise.
2- cos(x, /[, out, where, casting, order, ...]) 	Cosine element-wise.
3- tan(x, /[, out, where, casting, order, ...]) 	Compute tangent element-wise.
4- arcsin(x, /[, out, where, casting, order, ...]) 	Inverse sine, element-wise.
5- arccos(x, /[, out, where, casting, order, ...]) 	Trigonometric inverse cosine, element-wise.
6- arctan(x, /[, out, where, casting, order, ...]) 	Trigonometric inverse tangent, element-wise.
7- hypot(x1, x2, /[, out, where, casting, ...]) 	Given the “legs” of a right triangle, return its hypotenuse.
8- arctan2(x1, x2, /[, out, where, casting, ...]) 	Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
9- degrees(x, /[, out, where, casting, order, ...]) 	Convert angles from radians to degrees.
10- radians(x, /[, out, where, casting, order, ...]) 	Convert angles from degrees to radians.
11- unwrap(p[, discont, axis]) 	Unwrap by changing deltas between values to 2*pi complement.
12- deg2rad(x, /[, out, where, casting, order, ...]) 	Convert angles from degrees to radians.
13- rad2deg(x, /[, out, where, casting, order, ...]) 	Convert angles from radians to degrees.
14- Hyperbolic functions
15- sinh(x, /[, out, where, casting, order, ...]) 	Hyperbolic sine, element-wise.
16- cosh(x, /[, out, where, casting, order, ...]) 	Hyperbolic cosine, element-wise.
17- tanh(x, /[, out, where, casting, order, ...]) 	Compute hyperbolic tangent element-wise.
18- arcsinh(x, /[, out, where, casting, order, ...]) 	Inverse hyperbolic sine element-wise.
19- arccosh(x, /[, out, where, casting, order, ...]) 	Inverse hyperbolic cosine, element-wise.
20- arctanh(x, /[, out, where, casting, order, ...]) 	Inverse hyperbolic tangent element-wise.
21- Rounding
22- around(a[, decimals, out]) 	Evenly round to the given number of decimals.
23- round_(a[, decimals, out]) 	Round an array to the given number of decimals.
24- rint(x, /[, out, where, casting, order, ...]) 	Round elements of the array to the nearest integer.
25- fix(x[, out]) 	Round to nearest integer towards zero.
26- floor(x, /[, out, where, casting, order, ...]) 	Return the floor of the input, element-wise.
27- ceil(x, /[, out, where, casting, order, ...]) 	Return the ceiling of the input, element-wise.
28- trunc(x, /[, out, where, casting, order, ...]) 	Return the truncated value of the input, element-wise.
29- Sums, products, differences
30- prod(a[, axis, dtype, out, keepdims]) 	Return the product of array elements over a given axis.
31- sum(a[, axis, dtype, out, keepdims]) 	Sum of array elements over a given axis.
32- nanprod(a[, axis, dtype, out, keepdims]) 	Return the product of array elements over a given axis treating Not a Numbers (NaNs) as ones.
33- nansum(a[, axis, dtype, out, keepdims]) 	Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
34- cumprod(a[, axis, dtype, out]) 	Return the cumulative product of elements along a given axis.
35- cumsum(a[, axis, dtype, out]) 	Return the cumulative sum of the elements along a given axis.
36- nancumprod(a[, axis, dtype, out]) 	Return the cumulative product of array elements over a given axis treating Not a Numbers (NaNs) as one.
37- nancumsum(a[, axis, dtype, out]) 	Return the cumulative sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
38- diff(a[, n, axis]) 	Calculate the n-th discrete difference along given axis.
39- ediff1d(ary[, to_end, to_begin]) 	The differences between consecutive elements of an array.
40- gradient(f, *varargs, **kwargs) 	Return the gradient of an N-dimensional array.
41- cross(a, b[, axisa, axisb, axisc, axis]) 	Return the cross product of two (arrays of) vectors.
42- trapz(y[, x, dx, axis]) 	Integrate along the given axis using the composite trapezoidal rule.
43- Exponents and logarithms
44- exp(x, /[, out, where, casting, order, ...]) 	Calculate the exponential of all elements in the input array.
45- expm1(x, /[, out, where, casting, order, ...]) 	Calculate exp(x) - 1 for all elements in the array.
46- exp2(x, /[, out, where, casting, order, ...]) 	Calculate 2**p for all p in the input array.
47- log(x, /[, out, where, casting, order, ...]) 	Natural logarithm, element-wise.
48- log10(x, /[, out, where, casting, order, ...]) 	Return the base 10 logarithm of the input array, element-wise.
49- log2(x, /[, out, where, casting, order, ...]) 	Base-2 logarithm of x.
50- log1p(x, /[, out, where, casting, order, ...]) 	Return the natural logarithm of one plus the input array, element-wise.
51- logaddexp(x1, x2, /[, out, where, casting, ...]) 	Logarithm of the sum of exponentiations of the inputs.
52- logaddexp2(x1, x2, /[, out, where, casting, ...]) 	Logarithm of the sum of exponentiations of the inputs in base-2.
53- Other special functions
54- i0(x) 	Modified Bessel function of the first kind, order 0.
55- sinc(x) 	Return the sinc function.
56- Floating point routines
57- signbit(x, /[, out, where, casting, order, ...]) 	Returns element-wise True where signbit is set (less than zero).
58- copysign(x1, x2, /[, out, where, casting, ...]) 	Change the sign of x1 to that of x2, element-wise.
59- frexp(x[, out1, out2], / [[, out, where, ...]) 	Decompose the elements of x into mantissa and twos exponent.
60- ldexp(x1, x2, /[, out, where, casting, ...]) 	Returns x1 * 2**x2, element-wise.
61- nextafter(x1, x2, /[, out, where, casting, ...]) 	Return the next floating-point value after x1 towards x2, element-wise.
62- spacing(x, /[, out, where, casting, order, ...]) 	Return the distance between x and the nearest adjacent number.
63- Arithmetic operations
64- add(x1, x2, /[, out, where, casting, order, ...]) 	Add arguments element-wise.
65- reciprocal(x, /[, out, where, casting, ...]) 	Return the reciprocal of the argument, element-wise.
66- negative(x, /[, out, where, casting, order, ...]) 	Numerical negative, element-wise.
67- multiply(x1, x2, /[, out, where, casting, ...]) 	Multiply arguments element-wise.
68- divide(x1, x2, /[, out, where, casting, ...]) 	Divide arguments element-wise.
69- power(x1, x2, /[, out, where, casting, ...]) 	First array elements raised to powers from second array, element-wise.
70- subtract(x1, x2, /[, out, where, casting, ...]) 	Subtract arguments, element-wise.
71- true_divide(x1, x2, /[, out, where, ...]) 	Returns a true division of the inputs, element-wise.
72- floor_divide(x1, x2, /[, out, where, ...]) 	Return the largest integer smaller or equal to the division of the inputs.
73- float_power(x1, x2, /[, out, where, ...]) 	First array elements raised to powers from second array, element-wise.
74- fmod(x1, x2, /[, out, where, casting, ...]) 	Return the element-wise remainder of division.
75- mod(x1, x2, /[, out, where, casting, order, ...]) 	Return element-wise remainder of division.
76- modf(x[, out1, out2], / [[, out, where, ...]) 	Return the fractional and integral parts of an array, element-wise.
77- remainder(x1, x2, /[, out, where, casting, ...]) 	Return element-wise remainder of division.
78- divmod(x1, x2[, out1, out2], / [[, out, ...]) 	Return element-wise quotient and remainder simultaneously.
79- Handling complex numbers
80- angle(z[, deg]) 	Return the angle of the complex argument.
81- real(val) 	Return the real part of the complex argument.
82- imag(val) 	Return the imaginary part of the complex argument.
83- conj(x, /[, out, where, casting, order, ...]) 	Return the complex conjugate, element-wise.
84- Miscellaneous
85- convolve(a, v[, mode]) 	Returns the discrete, linear convolution of two one-dimensional sequences.
86- clip(a, a_min, a_max[, out]) 	Clip (limit) the values in an array.
87- sqrt(x, /[, out, where, casting, order, ...]) 	Return the positive square-root of an array, element-wise.
88- cbrt(x, /[, out, where, casting, order, ...]) 	Return the cube-root of an array, element-wise.
89- square(x, /[, out, where, casting, order, ...]) 	Return the element-wise square of the input.
90- absolute(x, /[, out, where, casting, order, ...]) 	Calculate the absolute value element-wise.
91- fabs(x, /[, out, where, casting, order, ...]) 	Compute the absolute values element-wise.
92- sign(x, /[, out, where, casting, order, ...]) 	Returns an element-wise indication of the sign of a number.
93- heaviside(x1, x2, /[, out, where, casting, ...]) 	Compute the Heaviside step function.
94- maximum(x1, x2, /[, out, where, casting, ...]) 	Element-wise maximum of array elements.
95- minimum(x1, x2, /[, out, where, casting, ...]) 	Element-wise minimum of array elements.
96- fmax(x1, x2, /[, out, where, casting, ...]) 	Element-wise maximum of array elements.
97- fmin(x1, x2, /[, out, where, casting, ...]) 	Element-wise minimum of array elements.
98- nan_to_num(x[, copy]) 	Replace nan with zero and inf with finite numbers.
99- real_if_close(a[, tol]) 	If complex input returns a real array if complex parts are close to zero.
100- interp(x, xp, fp[, left, right, period]) 	One-dimensional linear interpolation.'''
#numpy.reshape => Gives a new shape to an array without changing its data
x31 = np.arange(8).reshape(4,2) # 4*2 = 8 (number of elements)
print("x31 = ",x31)
print ('------------------------------')
x31a = np.arange(4).reshape(4) # reshape(4) => number of columns
print("x31a = ",x31a)
print ('------------------------------')
x32 = np.math.inf #IEEE 754 floating point representation of (positive) infinity.
print("x32 = ",x32)
print ('------------------------------')
#x33 = np.array([1]) / 0
#print("x33 = ",x33)
#print ('------------------------------')
x34 = np.math.nan #not a number
print("x34 = ",x34)
print ('------------------------------')

#Mask => Logic functions
'''
Logical operations:
logical_and(x1, x2, /[, out, where, ...]) 	Compute the truth value of x1 AND x2 element-wise.
logical_or(x1, x2, /[, out, where, casting, ...]) 	Compute the truth value of x1 OR x2 element-wise.
logical_not(x, /[, out, where, casting, ...]) 	Compute the truth value of NOT x element-wise.
logical_xor(x1, x2, /[, out, where, ...]) 	Compute the truth value of x1 XOR x2, element-wise.'''

x35 = np.arange(18).reshape(3,2,3) # 3*2*3 = 9 (number of elements)
print("x35 = ",x35)
print("x35.ndim = ",x35.ndim) # ndim = 3
print ('------------------------------')
x35a = np.arange(24).reshape(4,2,3)
print("x35a = ",x35a)
print("x35a.ndim = ",x35a.ndim) # ndim = 3
print ('------------------------------')
x35b = np.arange(40).reshape(2,5,4)
print("x35b = ",x35b)
print("x35b.ndim = ",x35b.ndim) # ndim = 3
print ('------------------------------')
x35c = np.arange(20).reshape(1,5,4)
print("x35c = ",x35c)
print("x35c.ndim = ",x35c.ndim) # ndim = 3
print ('------------------------------')
x36 = x35 < 4
print("x36 = ",x36)
print ('------------------------------')
x37 = x35 [x36]
print("x37 = ",x37)
print ('------------------------------')
x38 = np.logical_and(x35>=2 , x35<5)
print("x38 = ",x38)
print ('------------------------------')
x39 = x35 [x38]
print("x39 = ",x39)
print ('------------------------------')

#other
#NumPy.unique => Find the unique elements of an array
x40 = np.unique([1, 1, 2, 2, 3, 3,4,4,4,5])
print("x40 = ",x40)
print ('------------------------------')
x41 = np.unique(np.array(([[1, 1], [2, 2], [3,2], [4,5]]))) 
print("x41 = ",x41)
print ('------------------------------')
x42 = np.unique(np.array([[1, 2, 3], [1, 2, 3], [2, 3, 4]]), axis=0) #Return the unique rows of a 2D array
print("x42 = ",x42)
print ('------------------------------')
#numpy.union1d =>  Find the union of two arrays
a = np.arange(12,27,2).reshape(2,4)
print("a = ",a)
print ('------------------------------')
b = np.ones((2,3), dtype='1i')
print("b = ",b)
print ('------------------------------')
c = np.arange(1,7).reshape(3,2)
print("c = ",c)
print ('------------------------------')
x43 = np.union1d(b,c)
print("x43 = ",x43)
print ('------------------------------')
# find the union of more than two arrays, use functools.reduce
from functools import reduce
x44 = reduce(np.union1d, (a,b,c))
print("x44 = ",x44)
print ('------------------------------')
#numpy.intersect1d => Find the intersection of two arrays
x45 = np.intersect1d(b,c)
print("x45 = ",x45)
print ('------------------------------')
x46 = reduce(np.intersect1d, ([1, 3, 5, 3], [6, 1, 3, 1], [7, 3, 1, 2]))
print("x46 = ",x46)
print ('------------------------------')

# numpy.sort => Return a sorted copy of an array
x47 = np.array([[1,4,2,2],[3,1,4,5],[8,8,7,20]])
x48 = np.sort(x47)
print("x48 = ",x48)
print ('------------------------------')
x49 = np.sort(x47, axis=0)
print("x49 = ",x49)
print ('------------------------------')
x50 = np.sort(x47, axis=1)
print("x50 = ",x50)
print ('------------------------------')
x51 = np.sort(x47, axis=None)
print("x51 = ",x51)
print ('------------------------------')
#Using tuple for numpy.ndarray
x52 = np.array ((2,3,4,5))
print("x52 = ",x52)
print ('------------------------------')
x53 = np.array (((2,3,4,5),(5,6,8,7)))
print("x53 = ",x53)
print ('------------------------------')
print("type (x53) : ",type (x53))

#numpy.vstack => Stack arrays in sequence vertically (row wise)
x54 = np.array([2, 4, 6])
x55 = np.array([1, 3, 7])
x56 = np.vstack((x54,x55)) #The numbe of column (x54,x55) is same
#x56 = np.vstack([x54,x55])
print("x56 = ",x56)
print ('------------------------------')
x57 = np.vstack((x55,x54))
print("x57 = ",x57)
print ('------------------------------')
#numpy.hstack => Stack arrays in sequence horizontally (column wise)
x58 = np.hstack((x54,x55))
print("x58 = ",x58)

# numpy.ndarray and (for    in  )
x59 = np.array([[2,85,79,34],[5,78,36,44],[12,0,1,99]])
print('x59 = ')
for x in x59:   #show the rows separately
    print(x)
print ('------------------------------')
x60 = x59.reshape(12)
print('x60 = ')
for x in x60:
    print(x)
print ('------------------------------')
#numpy.ravel => Return a contiguous flattened array.
print('x59 = ')
for x in np.ravel(x59):  #for x in x60.ravel()
    print(x)
print ('------------------------------')
#Iterating Over Arrays
print('x59 = ')
for x in np.nditer(x59):
    print (x)
print ('------------------------------')
x61 = iter(x59)
for x in x59:
    print ('next(x61) = ',next(x61))
    print ('.............')
print ('------------------------------')
x62 = iter(x59)
for x in x59:
    y = next(x62)
    for z in y:
        print (z)        
        
x63 = np.mean (x59)
print("mean = ",x63)
print ('------------------------------')
x64 = np.var (x59)
print("var = ",x64)
print ('------------------------------')
x65 = np.std (x59)
print("std = ",x65)
print ('------------------------------')
x66 = np.median (x59)
print("median = ",x66)
print ('------------------------------')

#numpy.polyval(p, x) => Evaluate a polynomial at specific values
# ax^2+bx+c = 0
# 2x^2+x+3 = 0
x67 = np.array([2,1,3])
print ('x=2 => 2x^2+x+3 = 0, ', np.polyval(x67,2))
print ('------------------------------')
# numpy.polyder => Return the derivative of the specified order of a polynomial
print ('np.polyder(x67) : ', np.polyder(x67))
print ('------------------------------')
#numpy.polyint => Return an antiderivative (indefinite integral) of a polynomial
print ('np.polyint(x67) : ', np.polyint(x67))
print ('------------------------------')

# Comparing Lists and Arrays

# Make a list
L_1 = [1, 2, 3, 4]
# Make Equivalent Array
A_1 = np.array ([1, 2, 3, 4])

# print reverse of list
L_1.reverse()

#L_1 = L_1[::-1]

for e in L_1:
	print (e)

print ('-----------------')

# reverse the array
A_1 = A_1[::-1]

for i in A_1:
	print (i)


print ('-----------------')

# insert an element into list
L_1.append(5)
print (L_1)

print ('-----------------')

#A_1.append(5)   # Error
#print A_1

#A_1 = A_1 + [6,7] # eRROR
#print A_1  

#print ('-----------------')

L_1 = L_1 + [6,7]
print (L_1) 

print ('-----------------')

L_1 =  L_1 + L_1
print (L_1)

L_2 = []
for i in L_1:
	L_2.append(i+i)
print (L_2)

print ('-----------------')

A_1 = A_1+A_1
print (A_1)

print ('-----------------')

A_1 = A_1 * 2
print (A_1)

L_1 = L_1 * 2
print (L_1)

print ('-----------------')

L_1 = [1, 2, 3, 4]
#L_1 = L_1 **2 # Error
#print (L_1)

L_3 = []
for i in L_1:
	L_3.append(i * i)
print (L_3)

print ('-----------------')

A_1 = np.array ([1, 2, 3, 4])
A_1 = A_1 ** 2
print(A_1)

print ('-----------------')

print (np.sqrt(A_1))
print (np.log(A_1))
print (np.exp(A_1))

# for the list u need use a for loop
print ('-----------------')

x = np.array([1,2])
y = np.array([2,1])

dot = 0

for i, j in zip(x, y):
	dot += (i*j)
print(dot)

print (x*y)  # * ----> element by element multiplication

print (np.sum(x*y))
print ((x*y).sum())
print (np.dot(x,y))  #   dot ------> matrix multiplication
# angle between x, y
angle = np.arccos(x.dot(y)/(np.linalg.norm(x) * np.linalg.norm(y)))
print (angle)

x = np.array([[1,2,3],[4,5,6]])
y = np.array([[7,8],[9,10],[11,12]])

print(x.dot(y))
print (np.dot(x,y))
print(y.dot(x))
# print (x*y)
# print (np.sum(x*y))
# print ((x*y).sum())
print ('-----------------')

# Magnitude of the vector:  np.sqrt((x*x).sum())
print (np.linalg.norm(x))

print ('-----------------')

# list of lists
L = [[1, 2], [3, 4]]

A = np.array ([[4, 5], [6, 7]])


print(L[0])
print(L[0][0])
print(A[0][0])
print(A[0,0])

A_2 = np.matrix(L)
print(A_2)
A_3 = np.array(A_2)
print (A_3)

print ('-----------------')

# transpose

print (A_3.T)

# matrix is -----> two-dimensional numpy array
# vector is -----> one-dimensional array

print ('-----------------')

Z_a = np.zeros(14)
print (Z_a)

Z_a = np.zeros((14, 14))
print (Z_a)

Z_a = np.ones((14, 14))
print (Z_a)

Z_r = np.random.random ((14, 14))
print (Z_r)

#Z_r = np.random.randn ((14, 14))
#print (Z_r)

Z_r = np.random.randn (14, 14)
print (Z_r)

print (Z_r.mean())

print (Z_r.var())


print ('-----------------')

#inverse
A_4 = np.array([[2, 4], [1, 3]])
print (A_4)
A_4_inv = np.linalg.inv(A_4)
print(A_4_inv)
print (np.linalg.det(A_4))
print (np.diag(A_4))

# 1, 2 go in the diagonal and everything else is 0.
print (np.diag ([8, 9]))

# Compute the outer/inner product of two vectors
x = np.array([4, 8])
y = np.array([16, 32])
print (np.outer(x, y))
print (np.inner(x, y))
print (x.dot(y))
print (np.diag(A_4).sum())
print (np.trace(A_4))

print ('-----------------')

X_r = np.random.randn(100, 4)  # 100 samples and 4 features
cov = np.cov(X_r)
print (cov.shape)

# covariance <--------> transpose
cov = np.cov(X_r.T)   # 4 * 4
print (cov)

print ('-----------------')

#Covariance is a Symmetricmatrix
#eigenvalues, eigenvectors = np.eig(x) or np.eigh(x)
#eigh -------> symmetric and Hermitian matrix
#Symmetric means A = A**T
#Hermitian means A = A**H
#A**H = conjugate transpose of A

print (np.linalg.eigh(cov))
print (np.linalg.eig(cov))

print ('-----------------')
