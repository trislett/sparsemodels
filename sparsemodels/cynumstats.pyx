#cython: boundscheck=False, wraparound=False, nonecheck=False
from numpy cimport ndarray
import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
	double sqrt(double m)

def fast_correlation(x, y):
	x -= np.mean(x)
	y -= np.mean(y)
	x /= np.linalg.norm(x)
	y /= np.linalg.norm(y)
	return(np.dot(x, y))

def column_correlation(X, Y):
	xdim = X.shape[1]
	ydim = Y.shape[1]
	rarr = np.empty((xdim,ydim))
	for px in range(xdim):
		for py in range(ydim):
			rarr[px, py] = fast_correlation(X[:,px], Y[:,py])
	return(rarr)

def fast_std_dev(ndarray[np.float64_t, ndim=1] a not None):
	cdef Py_ssize_t i
	cdef Py_ssize_t n = a.shape[0]
	cdef double m = 0.0
	for i in range(n):
		m += a[i]
	m /= n
	cdef double v = 0.0
	for i in range(n):
		v += (a[i] - m)**2
	return sqrt(v / n)

def fast_se_of_slope(ndarray[np.float64_t, ndim=2] invXX not None,
						ndarray[np.float64_t, ndim=1] sigma2 not None):
	cdef int ny = len(sigma2)
	cdef int k = len(invXX[0])
	cdef int j
	cdef int f
	cdef np.ndarray se = np.zeros(shape=(k,ny), dtype=np.float64)
	for j in range(ny):
		for f in range(k):
			se[f,j] = sqrt(sigma2[j]*invXX[f,f])
	return se

def se_of_slope(num_voxel,invXX,sigma2, k):
	cdef int j
	cdef np.ndarray se = np.zeros(shape=(k,num_voxel), dtype=np.float32)
	for j in xrange(num_voxel):
		se[:,j] = np.sqrt(np.diag(sigma2[j]*invXX))
	return se

def fast_sigma_sqr(X, Y, a):
	return np.sum((Y - np.dot(X,a))**2, axis=0) / (X.shape[0] - X.shape[1])

def cy_lin_lstsqr_mat(X, y):
	return (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y)

def cy_lin_lstsqr_mat_residual(exog_vars, endog_arr):
	a = cy_lin_lstsqr_mat(exog_vars, endog_arr)
	resids = endog_arr - np.dot(exog_vars,a)
	return (a, np.sum(resids**2, axis=0))

def tval_fast(X, Y, invXX):
	a = cy_lin_lstsqr_mat(X, Y)
	sigma2 = fast_sigma_sqr(X, Y, a)
	se = fast_se_of_slope(invXX, sigma2)
	return a / se

def tval_original(X, invXX, y, n, k, numvoxel):
	a = cy_lin_lstsqr_mat(X, y)
	sigma2 = np.sum((y - np.dot(X,a))**2,axis=0) / (n - k)
	se = se_of_slope(numvoxel,invXX,sigma2,k)
	tvals = a / se
	return tvals


