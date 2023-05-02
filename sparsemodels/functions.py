#!/usr/bin/env python

import os
import sys
import itertools
import pickle
import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime
from random import choices

from scipy.stats import t as tdist
from scipy.stats import f as fdist
from scipy.stats import norm, chi2, pearsonr, gaussian_kde
from scipy.linalg import pinv
from statsmodels.stats.multitest import multipletests, fdrcorrection

from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
# suppress console because of weird permission around r
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore", category=RRuntimeWarning)
warnings.filterwarnings('ignore') 

# cython functions
from sparsemodels.cynumstats import cy_lin_lstsqr_mat

stats = importr('stats')
base = importr('base')
utils = importr('utils')

# autoinstalls the r packages... not the smartest thing to do.
# create an install R + packages script later
try:
	rgcca = importr('RGCCA')
except:
	utils.install_packages('RGCCA')
	rgcca = importr('RGCCA')

# General Functions

def generate_seeds(n_seeds, maxint = int(2**32 - 1)):
	return([np.random.randint(0, maxint) for i in range(n_seeds)])

def pickle_save_model(model, filename):
	pickle.dump(model, open(filename, 'wb'))

def pickle_load_model(filename):
	with open(filename, 'rb') as pfile:
		model = pickle.load(pfile)
	return(model)

def stack_ones(arr):
	"""
	Add a column of ones to an array
	
	Parameters
	----------
	arr : array

	Returns
	---------
	arr : array
		array with a column of ones
	
	"""
	return np.column_stack([np.ones(len(arr)),arr])

def dummy_code(variable, iscontinous = False, demean = True, unit_variance = False, weight_by_sqrt_numvar = False):
	"""
	Dummy codes a variable
	
	Parameters
	----------
	variable : array
		1D array variable of any type 

	Returns
	---------
	dummy_vars : array
		dummy coded array of shape [(# subjects), (unique variables - 1)]
	
	"""
	variable = np.array(variable)
	if iscontinous:
		if demean:
			variable = variable - np.mean(variable,0)
		if unit_variance:
			variable = variable / np.std(variable,0)
		dummy_vars = variable
	else:
		unique_vars = np.unique(variable)
		dummy_vars = []
		for var in unique_vars:
			temp_var = np.zeros((len(variable)))
			temp_var[variable == var] = 1
			dummy_vars.append(temp_var)
		dummy_vars = np.array(dummy_vars)[1:] # remove the first column as reference variable
		dummy_vars = np.squeeze(dummy_vars).astype(int).T
		if demean:
			dummy_vars = dummy_vars - np.mean(dummy_vars,0)
		if unit_variance:
			dummy_vars = dummy_vars / np.std(dummy_vars,0)
		if weight_by_sqrt_numvar:
			if dummy_vars.ndim > 1:
				dummy_vars = np.divide(dummy_vars, np.sqrt(dummy_vars.shape[1]))
	return dummy_vars

def gram_schmidt_orthonormalize(X, columns=True):
	"""
	Performs Gram-Schmidt orthogonalization on the input matrix X.
	If columns is True, the columns of X are orthogonalized; if False, the rows are.
	Returns the orthonormal matrix Q.
	"""
	if columns:
		Q, _ = np.linalg.qr(X)
	else:
		Q, _ = np.linalg.qr(X.T)
		Q = Q.T
	return(Q)

def orthonormal_projection(w):
	"""
	Finds the projection matrix onto the subspace spanned by the columns of w.
	Returns the orthonormal matrix representing the projection.
	"""
	return w.dot(np.linalg.inv(np.linalg.sqrtm(w.T.dot(w))))

def cumulative_sum_of_array(arr):
	"""
	Cumulatively sums values of an array and returns an array of the same shape.
	"""
	cumulative_arr = np.zeros((arr.shape))
	for i in range(len(arr)):
		cumulative_arr[i] = np.sum(arr[:(i+1)], axis = 0)
	return(cumulative_arr)

# model assessment functions
def r_score(y_true, y_pred, scale_data = True, multioutput = 'uniform_average'):
	"""
	Calculates the correlation between the true and predicted values.
	
	The same method used is this citation:
	Bilenko NY, Gallant JL. Pyrcca: Regularized Kernel Canonical Correlation Analysis in Python and Its Applications to Neuroimaging. Front Neuroinform. 2016 Nov 22;10:49. doi: 10.3389/fninf.2016.00049.
	"""
	
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	if scale_data:
		y_true = scale(y_true)
		y_pred = scale(y_pred)
	n_targets = y_true.shape[1]
	score = np.array([cy_lin_lstsqr_mat(y_true[:,target].reshape(-1,1), y_pred[:,target])[0] for target in range(n_targets)])
	if multioutput == 'uniform_average':
		score = np.mean(score)
	return(score)

def regression_metric_function(metric = 'r2_score', multioutput = 'uniform_average', custom_function = None):
	"""
	Passes the metric of choice to assess model accuracy, Internal functions are: ['r_score', 'r2_score', 'mean_squared_error', 'median_absolute_error'].
	custom_function allows passthrough of custom metric.
	"""
	assert multioutput in ['raw_values', 'uniform_average'], "Error: multioutput must be raw_values or uniform_average"
	if custom_function is None:
		if metric == 'r2_score':
			metric_function = r2_score
		elif metric == 'r_score':
			metric_function = r_score
		elif metric == 'median_absolute_error':
			metric_function = median_absolute_error
		elif metric == 'mean_squared_error':
			metric_function = mean_squared_error
		else:
			print("Metric [%s] is not a known function. Use set_function")
		return(metric_function)
	else:
		return(custom_function)


# Sparse Generalized Canonical Correlation Analysis for Multiblock Data
class sgcca_rwrapper:
	"""
	Wrapper class for the SGCCA function of the R package RGCCA.
	https://rdrr.io/cran/RGCCA/man/sgcca.html
	"""
	def __init__(self, design_matrix = None, l1_sparsity = None, n_comp = 1, scheme = "centroid", scale = True, init = "svd", bias = True, tol = sys.float_info.epsilon):
		"""
		Initialize the wrapper with hyperparameters for SGCCA.

		Parameters:
		-----------
		design_matrix : np.ndarray or None
			A matrix that specifies the structure of the data. 
			Default value is None, in which case the design matrix is set to (1 - identity(n_views)).
		l1_sparsity : np.ndarray or float or None
			A float or an array of floats that specifies the sparsity penalty on the outer weights.
			Default value is None, in which case the penalty is set to 1 for all views.
		n_comp : int or np.ndarray
			An integer or an array of integers that specifies the number of components for each view.
			Default value is 1 for all views. It is possible set a different number of components for each dataview
		scheme : str
			A string that specifies the algorithm used to solve the optimization problem.
			Scheme options are "horst", "factorial" or "centroid" (Default: "centroid")
			Horst scheme g(x) = x
				Penalizes structural negative correlation between components
			Centroid scheme g(x) = |x|
				Components can be negatively correlated
			Factorial scheme g(x) = x**2
				Components can be negatively correlated
		scale : bool
			A boolean that specifies whether to scale the views before running SGCCA.
			Default value is True.
		init : str
			A string that specifies the initialization method used to initialize the optimization problem.
			Default value is "svd".
		bias : bool
			A boolean that specifies whether to include a bias term in the optimization problem.
			Default value is True.
		tol : float
			A float that specifies the tolerance at which the model has converged. Currently set to float epsilon (smallest possible difference between floating-point numbers). 1e-12 may be more reasonable.

		Returns:
		--------
		None
		"""
		assert scheme in np.array(["horst", "factorial", "centroid"]), "Error: %s is not a valid scheme option. Must be: horst, factorial, or centroid" % scheme
		self.design_matrix = design_matrix
		self.l1_sparsity = l1_sparsity
		self.n_comp = n_comp
		self.scheme = scheme
		self.scale = scale
		self.init = init
		self.bias = bias
		self.penalty = "l1"
		self.tol = tol

	def scaleviews(self, views, centre = True, scale = True, div_sqrt_numvar = True, axis = 0):
		"""
		Helper function to center and scale the views.

		Parameters:
		-----------
		views : list of np.ndarrays
			A list of numpy arrays that represent the views.
		centre : bool
			A boolean that specifies whether to center the views.
			Default value is True.
		scale : bool
			A boolean that specifies whether to scale the views.
			Default value is True.
		div_sqrt_numvar : bool
			A boolean that specifies whether to divide the views by the square root of the number of variables.
			Default value is True.
		axis : int
			An integer that specifies the axis to use when computing the mean and standard deviation.

		Returns:
		--------
		scaled_views : list of np.ndarrays
			A list of numpy arrays that represent the scaled views.
		"""
		scaled_views = []
		for x in views:
			x_mean = np.mean(x, axis = axis)
			x_std = np.std(x, axis = axis)
			if centre:
				x = x - x_mean
			if scale:
				x = np.divide(x, x_std)
			if div_sqrt_numvar:
				x = np.divide(x, np.sqrt(x.shape[1]))
			scaled_views.append(x)
		return(list(scaled_views))

	def _rlist_to_nplist(self, robj):
		"""
		Converts an R list to a list of numpy arrays.
		"""
		len_robj = len(robj)
		arr_list = []
		for i in range(len_robj):
			arr_list.append(np.array(robj[i]))
		return(arr_list)

	def _final_crit(self, robj):
		"""
		Convert the r crit list an numpy array of the final values.
		"""
		len_robj = len(robj)
		final_values = np.zeros((len_robj))
		for i in range(len_robj):
			final_values[i] = np.array(robj[i])[-1]
		return(final_values)

	def check_sparsity(self, verbose = True):
		"""
		Checks if l1_sparsity is valid and adjusts it if necessary.
		"""
		if self.l1_sparsity.ndim == 1:
			self.l1_sparsity = self.l1_sparsity[np.newaxis,:]
		if self.l1_sparsity.ndim == 2:
			for v in range(self.n_views_):
				sthrehold = 1 / np.sqrt(self.views_[v].shape[1])
				sparsity = self.l1_sparsity[:,v]
				if np.any(sparsity < sthrehold):
					nsparsity = np.round(np.round(sthrehold, 4) + 0.0001, 4)
					if verbose:
						print("Sparsity of view[%d] is too low. Adjusting to new value = %1.4f" %(int(v), nsparsity))
					self.l1_sparsity[:,v] = nsparsity
		assert self.l1_sparsity.shape == (np.max(self.n_comp), self.n_views_), "Error: l1_sparsity shape (%d, %d) is not (%d, %d)" % (self.l1_sparsity.shape[0], self.l1_sparsity.shape[1], np.max(self.n_comp), self.n_views_)

	def fit(self, X, verbose = True):
		"""
		Fits the model for the given views.
		
		Parameters:
		-----------
		X: list of arrays or array-like
			The views to fit the model on.

		Returns:
		--------
		self: instance of sgcca_rwrapper
			Returns the instance of the class.
		"""
		self.views_ = X
		self.n_views_ = len(self.views_)
		# Default: complete design
		# eg. design_mat = np.zeros((8, 8))
		# design_mat[1:,:1] = 1
		# design_mat[:1,1:] = 1
		# set scheme = "factorial"
		if self.design_matrix is None:
			self.design_matrix = 1 - np.identity(self.n_views_)
		matidx = np.array(self.design_matrix, bool)
		matidx[np.tril_indices(self.n_views_)] = False
		self.matidx_ = matidx
		self.canonical_correlations_indicies_ = np.where(self.matidx_)

		# l1 contraints to the outer weights ranging from 1/sqrt(view[j].shape[1]) to 1. i.e., 1 / sqrt(nvars) per data view is the minimum sparsity.
		if self.l1_sparsity is None:
			self.l1_sparsity = np.repeat(1., self.n_views_)
		if np.isscalar(self.l1_sparsity):
			self.l1_sparsity = np.repeat(self.l1_sparsity, self.n_views_)
		if self.l1_sparsity.shape != (np.max(self.n_comp), self.n_views_):
			self.l1_sparsity = np.tile(self.l1_sparsity, np.max(self.n_comp)).reshape(np.max(self.n_comp), self.n_views_)
		if np.isscalar(self.n_comp):
			self.n_comp = np.repeat(self.n_comp, self.n_views_)
		if self.scale:
			self.views_ = self.scaleviews(self.views_)
		self.check_sparsity(verbose = verbose)
		
		numpy2ri.activate()
		fit = rgcca.rgcca(blocks = self.views_, 
							C = self.design_matrix,
							connection = self.l1_sparsity,
							ncomp = self.n_comp, 
							scheme = self.scheme,
							scale = False,
							method = str('rgcca'),
							init = self.init,
							bias = self.bias,
							tol = self.tol,
							verbose  = False)
		numpy2ri.deactivate()
		
		self.scores_ = np.array(fit.rx2('Y'))
		self.weights_outer_ = self._rlist_to_nplist(fit.rx2('a'))
		self.weights_ = self._rlist_to_nplist(fit.rx2('astar'))
		self.AVE_views_ = np.array(fit.rx2('AVE')[0]) # this is the mean of the structural coefficents
		self.AVE_outer_ = np.array(fit.rx2('AVE')[1])
		self.AVE_inner_ = np.array(fit.rx2('AVE')[2])
		if np.max(self.n_comp)== 1:
			self.crit = np.array(fit.rx2('crit'))
		else:
			self.crit = self._final_crit(fit.rx2('crit'))
		return(self)

	def transform(self, views, calculate_loading = False, outer = False):
		"""
		Transform input views into scores (canonical variates).

		Parameters:
		-----------
			views (list): A list of views to be transformed.
			calculate_loading (bool, optional): Whether to calculate the loadings or not. Defaults to False.
			outer (bool, optional): Whether to use outer weights or not. Defaults to False.

		Returns:
		--------
			scores: array(n_views, n_subjects, n_components)
				An array of transformed views
			(Optional) loadings: list (n_views)
				if calculate_loading is true, transformed views and a list of the loadings().
		"""
		assert len(views) == len(self.views_), "Error: The length of views and does not match model's number of views. transform_view can be used for a data view"
		if outer:
			weights = self.weights_outer_
		else:
			weights = self.weights_
		views = self.scaleviews(views)
		scores = []
		for v in range(self.n_views_):
			scores.append(np.dot(views[v], weights[v]))
		scores = np.array(scores)
		if calculate_loading:
			loadings = []
			for v in range(self.n_views_):
				vloadings = np.zeros((weights[v].shape))
				for c in range(self.n_comp[v]):
					vloadings[:,c] = cy_lin_lstsqr_mat(scale(scores[v][:,c]).reshape(-1,1), scale(views[v]))[0]
				loadings.append(vloadings)
			return(scores, loadings)
		else:
			return(scores)

	def transform_view(self, view, view_index):
		"""
		Transform input view into scores (canonical variate).
		
		Parameters:
		-----------
			view: np.ndarray 
				2d array with shape (n_variables, n_subjects)
			view_index: int
				The index of the data view of interest. e.g., view_index=0 is the first data view
		
		Returns:
		--------
			scores: array
				2d array scores with shape (n_subjects, n_components)
		"""
		assert view.shape[1] == self.views_[view_index].shape[1], "Error: the input view and model view_index must have the same number variables"
		X = self.scaleviews(view)
		scores = np.dot(X, self.weights_[view_index])
		scores = np.array(scores)
		return(scores)

	def transform_scores(self, view, scores):
		"""
		Transforms canonical variates scores to loadings using cython optimizated least square regression
		Returns loadings with the shape n_targets, n_components

		Parameters:
		-----------
			view: np.ndarray
				2d array with shape (n_variables, n_subjects)
			scores: array
				2d array scores with shape (n_subjects, n_components)
		Returns:
		--------
			loadings: array
				2d array scores with shape (n_subjects, n_components)
		"""
		n_components = scores.shape[1]
		n_targets = view.shape[1]
		loadings = np.zeros((n_targets, n_components))
		for c in range(n_components):
			loadings[:,c] = cy_lin_lstsqr_mat(scale(scores[:,c]).reshape(-1,1), scale(view))[0]
		return(loadings)

	def inverse_transform(self, scores, view_index):
		"""
		Transforms a score back to original space.

		Parameters:
		-----------
		scores : np.ndarray
			2d array with shape (n_subjects, n_components)
		view_index : int
			index of the view that was used to transform the scores to this component space

		Returns:
		--------
		proj : np.ndarray
			2d array with shape (n_subjects, n_variables)
			a projection of the scores back to the original space
		"""
		if hasattr(self, 'loadings_') is False:
			self.loadings_ = self.transform(self.views_, calculate_loading = True)[1]
		proj = np.matmul(scores, self.loadings_[view_index].T)
		proj *= np.std(self.views_[view_index], 0)
		proj += np.std(self.views_[view_index], 0)
		return(proj)

	def predict(self, scores, response_index, verbose = False):
		"""
		Score prediction based on a linear regression model with one view as the response variable.

		Parameters:
		-----------
		scores : np.ndarray
			3d array with shape (n_views, n_subjects, n_components)
		response_index : int
			index of the view in scores that is used as the response variable in the linear regression model
		verbose : bool, optional
			boolean flag that, if set to True, prints out the R-squared score for each component

		Returns:
		--------
		yhat : np.ndarray
			3d array with shape (n_views, n_subjects, n_components)
			a prediction of the score based on a linear regression model with one view as the response variable
		"""
		if scores.ndim == 2:
			scores = scores[:,:,np.newaxis]
		n_view, n_subjects, n_comps = scores.shape
		independent_index = np.arange(0, n_view, 1)
		independent_index = independent_index[independent_index!=response_index]
		yhat = np.zeros((scores[response_index,:,:].shape))
		for i in range(scores.shape[2]):
			X_ = scores[independent_index,:,i].T
			Y_ = scores[response_index,:,i].T
			reg = LinearRegression(fit_intercept=False).fit(X_,Y_)
			if verbose:
				R2score = reg.score(X_,Y_)
				print("Component [%d] R2(score) = %1.3f" % (int(i+1), R2score))
			yhat[:, i] = reg.predict(X_)
		return(yhat)

	def calculate_average_variance_explained(self, views, use_gram_schmidt_orthonormalize = False):
		"""
		Computes the average variance explained (AVE) for a given set of views.
		AVE is a measure of the proportion of variance of the original data explained
		by a set of components. It is used to evaluate the quality of a multi-view model.
		
		Parameters:
		-----------
		views : list of arrays, shape = [n_views, n_samples, n_features]
			The views for which to compute AVE.
		use_gram_schmidt_orthonormalize : bool, optional (default=False)
			Whether to orthonormalize the scores if they are correlated.
		
		Returns:
		--------
		AVE_views_ : array, shape = [n_views, n_components]
			The AVE of each view.
		AVE_outer_ : array, shape = [n_components]
			The AVE of the outer model, which is a global indicator of model quality.
		AVE_innermodel : array, shape = [n_components]
			The AVE of the inner model, which is the average correlation between blocks.
		"""
		assert len(views) == len(self.views_), "Error: The length of views and does not match model's number of views."
		scores, loadings = self.transform(views, calculate_loading=True, outer=False)
		n_comp = np.max(self.n_comp)
		# Orthonormalize the scores if they are correlated.
		# Q = gram_schmidt_orthonormalize(score)
		# I = Q.T@Q
		# np.allclose(np.identity(score.shape[1]), np.round(I,8))
		if use_gram_schmidt_orthonormalize:
			for v in range(self.n_views_):
				scores[v] = gram_schmidt_orthonormalize(scores[v])
		
		# Compute AVE for each view
		AVE_views_ = np.zeros((self.n_views_, n_comp))
		for v in range(self.n_views_):
			AVE_views_[v] = np.mean((np.square(loadings[v])), 0)
		
		# Compute AVE for the outer model, which is a global indicator of model quality
		n_vars_ = np.zeros((self.n_views_))
		for v in range(self.n_views_):
			n_vars_[v] = views[v].shape[1]
		AVE_outer_ = np.zeros((self.n_views_, n_comp))
		for v in range(self.n_views_):
			AVE_outer_[v] = np.mean((np.square(loadings[v])), 0) * n_vars_[v]
		AVE_outer_ /= n_vars_.sum()
		AVE_outer_ = AVE_outer_.sum(0)
		
		# Compute AVE for the inner model, which is the average correlation between blocks
		canonical_correlation = np.zeros((n_comp, len(self.canonical_correlations_indicies_[0])))
		for c in range(n_comp):
			canonical_correlation[c] = np.corrcoef(scores[:,:,c])[self.canonical_correlations_indicies_]
		AVE_innermodel = (canonical_correlation**2).mean(1)
		
		return(AVE_views_, AVE_outer_, AVE_innermodel)

	def _covariance_criteria(self, scores, bias=True):
		"""
		Helper function to outputs the final criteria from scores. This is useful for cross-validation.
		"""
		_, n, n_comp = scores.shape
		if bias:
			b = np.divide((n-1), n)
		else:
			b = 1
		crit = np.zeros((n_comp))
		for c in range(n_comp):
			if self.scheme == 'centroid':
				crit[c] = b * np.sum(np.abs(np.cov(scores[:,:,c]))[self.design_matrix == 1])
			elif self.scheme == 'factorial':
				crit[c] = b * np.sum(np.square(np.cov(scores[:,:,c]))[self.design_matrix == 1])
			else:
				crit[c] = b * np.sum(np.cov(scores[:,:,c])[self.design_matrix == 1])
		return(crit)

class parallel_sgcca():
	def __init__(self, n_jobs = 12, design_matrix = None, scheme = "centroid", n_permutations = 10000):
		"""
		Main SGCCA function
		"""
		self.n_jobs = n_jobs
		self.n_permutations = n_permutations
		self.design_matrix = design_matrix
		assert scheme in np.array(["horst", "factorial", "centroid"]), "Error: %s is not a valid scheme option. Must be: horst, factorial, or centroid" % scheme
		self.scheme = scheme

	def _check_design_matrix(self, n_views):
		if self.design_matrix is None:
			self.design_matrix = 1 - np.identity(n_views)
			matidx = np.array(self.design_matrix, bool)
			matidx[np.tril_indices(n_views)] = False
			self.matidx_ = matidx

	def _datestamp(self):
		print("2023_24_04")

	def nfoldsplit_group(self, group, n_fold = 10, holdout = 0, train_index = None, verbose = False, debug_verbose = False, seed = None):
		"""
		Creates indexed array(s) for k-fold cross validation with holdout option for test data. The ratio of the groups are maintained. To reshuffle the training, if can be passed back through via index_train.
		The indices are always based on the original grouping variable. i.e., the orignal data.
		
		Parameters
		----------
		group : array
			List array with length of number of subjects. 
		n_fold : int
			The number of folds
		holdout : float
			The amount of data to holdout ranging from 0 to <1. A reasonable holdout is around 0.3 or 30 percent. If holdout = None, then returns test_index = None. (default = 0)
		train_index : array
			Indexed array of training data. Holdout must be zero (holdout = 0). It is useful for re-shuffling the fold indices or changing the number of folds.
		verbose : bool
			Prints out the splits and some basic information
		debug_verbose: bool
			Prints out the indices by group
		Returns
		---------
		train_index : array
			index array of training data
		fold_indices : object
			the index array for each fold (n_folds, train_fold_size)
		test_index : array or None
			index array of test data
		"""
		if seed is None:
			np.random.seed(np.random.randint(4294967295))
		else:
			np.random.seed(seed)
		test_index = None
		original_group = group[:]
		ugroup = np.unique(group)
		lengroup = len(group)
		indices = np.arange(0,lengroup,1)
		if holdout != 0:
			assert holdout < 1., "Error: Holdout ratio must be >0 and <1.0. Try .3"
			assert train_index is None, "Error: train index already exists."
			indx_0 = []
			indx_1 = []
			for g in ugroup:
				pg = np.random.permutation(indices[group==g])
				indx_0.append(pg[:int(len(pg)*holdout)])
				indx_1.append(pg[int(len(pg)*holdout):])
			train_index = np.concatenate(indx_1)
			test_index = np.concatenate(indx_0)
			group = group[train_index]
			if verbose:
				print("Train data size = %s, Test data size = %s [holdout = %1.2f]" %(len(train_index), len(test_index), holdout))
		else:
			if train_index is None:
				train_index = indices[:]
			else:
				group = group[train_index]
		# reshuffle for good luck
		gsize = []
		shuffle_train = []
		for g in ugroup:
			pg = np.random.permutation(train_index[group==g])
			gsize.append(len(pg))
			shuffle_train.append(pg)
		train_index = np.concatenate(shuffle_train)
		group = original_group[train_index]
		split_sizes = np.divide(gsize, n_fold).astype(int)
		if verbose:
			for s in range(len(ugroup)):
				print("Training group [%s]: size n=%d, split size = %d, remainder = %d" % (ugroup[s], gsize[s], split_sizes[s], int(gsize[s] % split_sizes[s])))
			if test_index is not None:
				for s in range(len(ugroup)):
					original_group[test_index] == ugroup[s]
					test_size = np.sum((original_group[test_index] == ugroup[s])*1)
					print("Test group [%s]: size n=%d, holdout percentage = %1.2f" % (ugroup[s], test_size, np.divide(test_size * 100, test_size+gsize[s])))
		fold_indices = []
		for n in range(n_fold):
			temp_index = []
			for i, g in enumerate(ugroup):
				temp = train_index[group==g]
				if n == n_fold-1:
					temp_index.append(temp[n*split_sizes[i]:])
				else:
					temp_index.append(temp[n*split_sizes[i]:((n+1)*split_sizes[i])])
				if debug_verbose:
					print(n)
					print(g)
					print(original_group[temp_index[-1]])
					print(temp_index[-1])
			fold_indices.append(np.concatenate(temp_index))
		train_index = np.sort(train_index)
		fold_indices = np.array(fold_indices, dtype = object)
		if holdout != 0:
			test_index = np.sort(test_index)
		if verbose:
			for i in range(n_fold):
				print("\nFOLD %d:" % (i+1))
				print(np.sort(original_group[fold_indices[i]]))
			if test_index is not None:
				print("\nTEST:" )
				print(np.sort(original_group[test_index]))
		return(fold_indices, train_index, test_index)

	def create_nfold(self, group, n_fold = 10, holdout = 0.3, verbose = False):
		"""
		Imports the data and runs nfoldsplit_group.
		"""
		fold_indices, train_index, test_index  = self.nfoldsplit_group(group = group,
																							n_fold = n_fold,
																							holdout = holdout,
																							train_index = None,
																							verbose = verbose,
																							debug_verbose = False)
		self.train_index_ = train_index
		self.fold_indices_ = fold_indices
		self.test_index_ = test_index
		self.group_ = group

	def subsetviews(self, views, indices):
		"""
		Subsets the views data for each view using the given indices.

		Parameters:
		-----------
		views : list
			A list of np.ndarray views data
		indices : np.ndarray
			1d array of indices to use for subsetting the views data

		Returns:
		--------
		subsetdata : list
			A list of np.ndarray views data
			each element in the list corresponds to a view from the input list
		"""
		subsetdata = []
		for v in range(len(views)):
			subsetdata.append(views[v][indices])
		return(subsetdata)

	def bootstrap_views(self, views, seed = None):
		"""
		Bootstraps with replacement the rows of each view in the input list of views (or scores).

		Parameters:
		-----------
		views : list
			A list of np.ndarray views data to permute.
		seed : int, optional
			Seed for the random number generator. Default is None.

		Returns:
		--------
		permutedviews : list
			A list of np.ndarray views data with permuted rows.
			Each element in the list corresponds to a view from the input list.
		"""
		if seed is None:
			np.random.seed(np.random.randint(4294967295))
		else:
			np.random.seed(seed)
		
		n = len(views[0])
		indices = np.random.choice(n, size=n, replace=True)
		bsviews = []
		for v in range(len(views)):
			bsviews.append(views[v][indices])
		return(bsviews)

	def permute_views(self, views, seed = None):
		"""
		Randomly permutes the rows of each view in the input list of views (or scores).

		Parameters:
		-----------
		views : list
			A list of np.ndarray views data to permute.
		seed : int, optional
			Seed for the random number generator. Default is None.

		Returns:
		--------
		permutedviews : list
			A list of np.ndarray views data with permuted rows.
			Each element in the list corresponds to a view from the input list.
		"""
		if seed is None:
			np.random.seed(np.random.randint(4294967295))
		else:
			np.random.seed(seed)
		permutedviews = []
		for v in range(len(views)):
			permutedviews.append(np.random.permutation(views[v]))
		return(permutedviews)

	def _prediction_mccv(self, p, views, train_index, l1, split, n_comp = 1, metric_function = regression_metric_function(metric = 'r2_score'), tol = 1e-3, seed = None):
		"""
		Perform a single iteration of Monte Carlo cross-validation (MCCV).

		Parameters:
		-----------
		p : int
			Index of the current permutation.
		views : list of arrays
			List of views.
		train_index : array
			Indices for the training set.
		l1 : float
			L1 sparsity parameter for sGCCA.
		split : int
			Size of the test set.
		n_comp : int, optional
			Number of components to extract. Default is 1.
		metric_function : function, optional
			Function for computing the evaluation metric. Default is r2_score.
		tol : float, optional
			Tolerance for stopping criterion. Default is 1e-3.
		seed : int, optional
			Seed for random number generator. Default is None.

		Returns:
		--------
		float
		The evaluation metric score for the current permutation.
		"""
		def _mccvsplit(indices, split, seed = None):
			"""
			Helper function to split indices for Monte Carlo cross-validation.

			Parameters:
			-----------
			indices : array
				Array of indices to split.
			split : int
				Size of the test set.
			seed : int, optional
				Seed for random number generator. Default is None.

			Returns:
			--------
			array, array
				Training and test indices.
			"""
			if seed is None:
				np.random.seed(np.random.randint(4294967295))
			else:
				np.random.seed(seed)
			rand_indices = np.random.permutation(indices)
			return(np.sort(rand_indices[:-split]), np.sort(rand_indices[-split:]))
		cvtrainidx, cvtestidx = _mccvsplit(train_index, split, seed = seed)
		mtrain = self.subsetviews(views, cvtrainidx)
		mtest = self.subsetviews(views, cvtestidx)
		mfit = sgcca_rwrapper(design_matrix = self.design_matrix,
									l1_sparsity = l1,
									n_comp = n_comp,
									scheme = self.scheme,
									tol = tol).fit(mtrain)
		mscoretest = mfit.transform(mtest)
		return(r2_score(mscoretest[0], mfit.predict(mscoretest, 0, verbose = False)))

	def prediction_mccv(self, views, l1_range = np.arange(0.1,1.1,.1),n_component_range = np.arange(1,11,1), n_perm_per_block = 200, split_test_ratio = 0.2, metric_function = regression_metric_function(metric = 'r2_score'), tol = 1e-3):
		"""
		Montecarlo cross-validation

		Parameters:
		-----------
		views : list of numpy arrays
			List of views for each subject. Each numpy array should have shape (n_samples, n_features).
		l1_range : numpy array, default=np.arange(0.1,1.1,.1)
			Array of L1 regularization values to test.
		n_component_range : numpy array, default=np.arange(1,11,1)
			Array of SGCCA component values to test.
		n_perm_per_block : int, default=200
			Number of permutations per block.
		split_test_ratio : float, default=0.2
			Ratio of subjects to hold out for testing.
		metric_function : function, default=regression_metric_function(metric='r2_score')
			Metric function to evaluate the model. Default is 'r2_score'.
		tol : float, default=1e-3
			Tolerance for convergence.
		
		Returns:
		--------
		stat_mean : np.ndarray
			Array of mean metric values across all permutations, with shape (len(n_component_range), len(l1_range)).
		stat_std : np.ndarray
			Array of standard deviation of metric values across all permutations, with shape (len(n_component_range), len(l1_range)).
		""" 
		assert hasattr(self,'train_index_'), "Error: run create_nfold"
		self._check_design_matrix(len(views))
		
		# For the future, use groups.
		
		views_train = self.subsetviews(views, self.train_index_)
		n_subs = views_train[0].shape[0]
		split = int(n_subs*split_test_ratio)
		stat_mean = np.zeros((len(l1_range), len(n_component_range)))
		stat_std = np.zeros((len(l1_range), len(n_component_range)))
		for c, n_comp in enumerate(n_component_range):
			for i, l1 in enumerate(l1_range):
				seeds = generate_seeds(n_perm_per_block)
				stat = Parallel(n_jobs = self.n_jobs, backend='multiprocessing')(
							delayed(self._prediction_mccv)(p, views = views,
																design_matrix = self.design_matrix,
																train_index = self.train_index_,
																l1 = l1,
																split = split,
																n_comp = n_comp,
																metric_function = metric_function,
																tol = tol,
																seed = seeds[p]) for p in tqdm(range(n_perm_per_block)))
				stat_mean[c, i] = np.mean(stat)
				stat_std[c, i] = np.std(stat)
		return(stat_mean, stat_std)

	def _premute_model(self, p, views_train, l1, views_test = None, aggregate_values = True, n_comp = 1, metric = 'objective_function', tol = 1e-3, seed = None):
		"""
		Permutation testing for model significance and hyperparameter selection.
		"""
		if seed is None:
			np.random.seed(np.random.randint(4294967295))
		else:
			np.random.seed(seed)
		for attempt in range(10):
			try:
				pmdl = sgcca_rwrapper(design_matrix = self.design_matrix,
											l1_sparsity = l1,
											n_comp = n_comp,
											scheme = self.scheme,
											scale = True,
											init = "svd",
											bias = True,
											tol = tol).fit(self.permute_views(views_train), verbose = False)
			except:
				print("Error in permuted model. Reshuffling try %d/10" % (attempt+1))
				np.random.seed(seed+1)
				pmdl = sgcca_rwrapper(design_matrix = self.design_matrix,
											l1_sparsity = l1,
											n_comp = n_comp,
											scheme = self.scheme,
											scale = True,
											init = "svd",
											bias = True,
											tol = tol).fit(self.permute_views(views_train), verbose = False)
			else:
				break
		if metric == 'fisherz_transformation':
			pstat = np.zeros((n_comp))
			for c in range(n_comp):
				mat = np.corrcoef(pmdl.scores_[:,:,c])
				pstat[c] = np.sum(np.abs(np.arctanh(mat))[self.matidx_])
			if aggregate_values:
				pstat = np.sum(pstat)
		elif metric == 'mean_correlation':
			pstat = np.zeros((n_comp))
			for c in range(n_comp):
				mat = np.corrcoef(pmdl.scores_[:,:,c])
				pstat[c] = np.mean(np.abs(mat)[self.matidx_])
			if aggregate_values:
				pstat = np.mean(pstat)
		elif metric == 'AVE_inner':
			pstat = pmdl.AVE_inner_
			if aggregate_values:
				pstat = np.mean(pmdl.AVE_inner_)
		else:
			pstat = pmdl.crit
			if aggregate_values:
				pstat = np.sum(pmdl.crit)
		if views_test is None:
			return(pstat)
		else:
			scores_test = pmdl.transform(views_test)
			pstat_test = np.zeros((n_comp))
			if metric == 'fisherz_transformation':
				for c in range(n_comp):
					mat = np.corrcoef(scores_test[:,:,0])
					pstat_test[c] = np.sum(np.abs(np.arctanh(mat))[self.matidx_])
				if aggregate_values:
					pstat_test = np.sum(pstat_test)
			elif metric == 'mean_correlation':
				for c in range(n_comp):
					mat = np.corrcoef(scores_test[:,:,c])
					pstat_test[c] = np.mean(np.abs(mat)[self.matidx_])
				if aggregate_values:
					pstat_test = np.mean(pstat_test)
			elif metric == 'AVE_inner':
				pstat_test = pmdl._covariance_criteria(scores_test)
				if aggregate_values:
					pstat_test = np.sum(pstat_test)
			else:
				pstat_test = pmdl.calculate_average_variance_explained(views_test)[2]
				if aggregate_values:
					pstat_test = np.mean(pstat_test)
			return(pstat, pstat_test)

	def run_parallel_permute_model(self, metric = 'objective_function', tol = 1e-3, save_permutations = True):
		"""
		Parameters
		----------
		metric: str
			Metric options are: objective_function, AVE_inner, fisherz_transformation, or mean_correlation. (Default: 'objective_function')
		tol: float
			tolerance of the model
		Returns
		---------
			self
		"""
		assert hasattr(self,'model_obj_'), "Error: run fit_model"
		n_comp = np.max(self.n_components_)
		mdl = sgcca_rwrapper(design_matrix = self.design_matrix,
									l1_sparsity = self.l1_sparsity_,
									n_comp = self.n_components_,
									scheme = self.scheme,
									scale = True,
									init = "svd",
									bias = True,
									tol = tol).fit(self.views_train_)
		test_scores = mdl.transform(self.views_test_)
		stat_train = np.zeros((n_comp))
		stat_test = np.zeros((n_comp))
		if metric == 'fisherz_transformation':
			for c in range(n_comp):
				mat = np.corrcoef(mdl.scores_[:,:,c])
				stat_train[c] = np.sum(np.abs(np.arctanh(mat))[mdl.matidx_])
				mat = np.corrcoef(test_scores[:,:,c])
				stat_test[c] = np.sum(np.abs(np.arctanh(mat))[mdl.matidx_])
		elif metric == 'mean_correlation':
			for c in range(n_comp):
				mat = np.corrcoef(mdl.scores_[:,:,c])
				stat_train[c] = np.mean(np.abs(mat)[self.matidx_])
				mat = np.corrcoef(test_scores[:,:,c])
				stat_test[c] = np.mean(np.abs(mat)[self.matidx_])
		elif metric == 'AVE_inner':
			stat_train[:] = mdl.AVE_inner_
			stat_test[:] = mdl.calculate_average_variance_explained(self.views_test_)[2]
		else:
			stat_train[:] = np.array(mdl.crit)
			stat_test[:] = mdl._covariance_criteria(test_scores)
		seeds = generate_seeds(self.n_permutations)
		output = Parallel(n_jobs = self.n_jobs, backend='multiprocessing')(
					delayed(self._premute_model)(p = p,
														views_train = self.views_train_,
														l1 = self.l1_sparsity_,
														views_test = self.views_test_,
														n_comp = self.n_components_,
														aggregate_values = False,
														metric = metric,
														tol = tol,
														seed = seeds[p]) for p in tqdm(range(self.n_permutations)))
		statstar_train, statstar_test = zip(*output)
		statstar_train = np.array(statstar_train)
		if statstar_train.ndim == 1:
			statstar_train = statstar_train[:,np.newaxis]
		statstar_test = np.array(statstar_test)
		if statstar_test.ndim == 1:
			statstar_test = statstar_test[:,np.newaxis]
		zstat_train = (stat_train - statstar_train.mean(0)) / statstar_train.std(0)
		zstat_test = (stat_test - statstar_test.mean(0)) / statstar_test.std(0)
		stat_train_p = np.zeros((np.max(self.n_components_)))
		stat_test_p = np.zeros((np.max(self.n_components_)))
		for c in range(self.n_components_):
			stat_train_p[c] = np.divide(np.searchsorted(np.sort(statstar_train[:,c]), stat_train[c]), len(self.n_permutations))
			stat_test_p[c] = np.divide(np.searchsorted(np.sort(statstar_test[:,c]), stat_test[c]), len(self.n_permutations))
		# save permuted models
		self.perm_stat_train_ = stat_train
		self.perm_stat_train_z_ = zstat_train
		self.perm_stat_train_p_ = stat_train_p
		self.perm_stat_test_ = stat_test
		self.perm_stat_test_z_ = zstat_test
		self.perm_stat_test_p_ = stat_test_p
		if save_permutations:
			self.perm_statstar_train_ = statstar_train
			self.perm_statstar_test_ = statstar_test

	def run_parallel_parameterselection(self, views, l1_range = np.arange(0.1,1.1,.1), n_perm_per_block = 200, metric = 'objective_function', tol = 1e-3, verbose = True):
		"""
		Parameters
		----------
		metric: str
			Metric options are: objective_function, fisherz_transformation, or mean_correlation. (Default: 'objective_function')
		view_index: None or int
			Sets the view to optimize. Must be set of for prediction. If None, all pairwise correlations are used.
		Returns
		---------
			self
		"""
		assert hasattr(self,'train_index_'), "Error: run create_nfold"
		self._check_design_matrix(len(views))
		
		views_train = self.subsetviews(views, self.train_index_)
		zstat = np.zeros_like(l1_range)
		tmetric = np.zeros_like(l1_range)
		tstar_blocks = np.zeros((len(l1_range), n_perm_per_block))
		parameterselection_l1_penalties = []
		for i, l1 in enumerate(l1_range):
			mdl = sgcca_rwrapper(design_matrix = self.design_matrix,
										l1_sparsity = l1,
										n_comp = 1,
										scheme = self.scheme,
										scale = True,
										init = "svd",
										bias = True,
										tol = tol).fit(views_train)
			parameterselection_l1_penalties.append(mdl.l1_sparsity)
			mat = np.corrcoef(mdl.scores_[:,:,0])
			if metric == 'fisherz_transformation':
				t = np.sum(np.abs(np.arctanh(mat))[self.matidx_])
			elif metric == 'mean_correlation':
				t = np.mean(np.abs(mat)[self.matidx_])
			else:
				t = np.sum(mdl.crit)
			seeds = generate_seeds(n_perm_per_block)
			tstar = Parallel(n_jobs = self.n_jobs, backend='multiprocessing')(
						delayed(self._premute_model)(p = p,
															views_train = views_train,
															l1 = l1,
															n_comp = 1,
															metric = metric,
															tol = tol,
															seed = seeds[p]) for p in tqdm(range(n_perm_per_block)))
			z = np.divide((t - np.mean(tstar)), np.std(tstar))
			tmetric[i] = t
			zstat[i] = z
			tstar_blocks[i] = tstar
			if verbose:
				print("Sparsity [%1.2f]: t = %1.5f, mean(t*) = %1.5f, std(t*) = %1.5f, z-stat = %1.5f" % (l1, t, np.mean(tstar), np.std(tstar), z))
			self.parameterselection_zstat_ = zstat
			self.parameterselection_tmetric_ = tmetric
			self.parameterselection_tstar_ = tstar_blocks
			self.parameterselection_besttuningindex_ = np.argmax(zstat)
			self.parameterselection_bestpenalties_ = l1_range[self.parameterselection_besttuningindex_]
			self.parameterselection_l1penalties_ = np.array(parameterselection_l1_penalties)

	def fit_model(self, views, n_components, l1_sparsity = None):
		"""
		Fits SGCCA model to the provided views.

		Parameters:
		-----------
		views : list of np.ndarray
			A list of 2D arrays, where each array represents a view.
			The shape of each view is (n_variables, n_subjects).
		n_components : int or np.ndarray
			Number of components to fit.
		l1_sparsity : float or np.ndarray, default=None
			L1 sparsity parameter for the model. If None, the best
			penalty parameter found during parameter selection (if
			performed) will be used. If neither parameter selection nor
			l1_sparsity is specified, l1_sparsity is set to 1.0.

		Returns:
		--------
		self : object
			The fitted SGCCA model object, containing the following
			attributes:
			- views_ : list of np.ndarray
				The input views.
			- n_views_ : int
				The number of input views.
			- design_matrix : np.ndarray
				The concatenated design matrix.
			- views_train_ : list of np.ndarray
				The training set views.
			- views_test_ : list of np.ndarray
				The test set views.
			- train_scores_ : np.ndarray
				The SGCCA scores of the training set.
				Shape: (n_subjects, n_components).
			- train_loadings_ : list of np.ndarray
				The SGCCA loadings of the training set views.
				Shape: (n_variables, n_components).
			- test_scores_ : np.ndarray
				The SGCCA scores of the test set.
				Shape: (n_subjects, n_components).
			- test_loadings_ : list of np.ndarray
				The SGCCA loadings of the test set views.
				Shape: (n_variables, n_components).
			- canonical_correlations_indicies_ : tuple
				A tuple of two arrays representing the row and column
				indices of the canonical correlations in the full
				correlation matrix.
			- train_canonical_correlations_ : np.ndarray
				The canonical correlations between training set scores.
				Shape: (n_components, n_pairwise_comparisons).
			- test_canonical_correlations_ : np.ndarray
				The canonical correlations between test set scores.
				Shape: (n_components, n_pairwise_comparisons).
			- n_components_ : int
				The number of components in the model.
			- l1_sparsity_ : float
				The l1 sparsity parameter used in the model.
			- model_obj_ : sgcca_rwrapper object
				The underlying SGCCA model object.
		"""
		assert hasattr(self,'fold_indices_'), "Error: run create_nfold"
		self.views_ = views
		self.n_views_ = len(views)
		self._check_design_matrix(self.n_views_)

		if l1_sparsity is None:
			if hasattr(self,'parameterselection_bestpenalties_'):
				l1_sparsity = self.parameterselection_bestpenalties_
				print("Parameter selection detected. Setting l1 sparsity to: %1.2f for all views" % l1_sparsity)
			else:
				l1_sparsity = 1.
				print("Setting l1 sparsity to 1.0 for all views")

		# note: scaling is done internally by sgcca_wrapper
		self.views_train_ = self.subsetviews(views, self.train_index_)
		self.views_test_ = self.subsetviews(views, self.test_index_)

		mdl = sgcca_rwrapper(design_matrix = self.design_matrix,
									l1_sparsity = l1_sparsity,
									n_comp = n_components,
									scheme = self.scheme).fit(self.views_train_)
		train_scores_, train_loadings_ = mdl.transform(self.views_train_, calculate_loading = True)
		self.train_scores_ = train_scores_
		self.train_loadings_ = train_loadings_
		test_scores_, test_loadings_ = mdl.transform(self.views_test_, calculate_loading = True)
		self.test_scores_ = test_scores_
		self.test_loadings_ = test_loadings_
		corr_index = np.where(self.matidx_)
		train_canonical_correlation = np.zeros((np.max(n_components), len(corr_index[0])))
		test_canonical_correlation = np.zeros((np.max(n_components), len(corr_index[0])))
		for c in range(np.max(n_components)):
			train_canonical_correlation[c] = np.corrcoef(self.train_scores_[:,:,c])[corr_index]
			test_canonical_correlation[c] = np.corrcoef(self.test_scores_[:,:,c])[corr_index]

		selected_variables_ = []
		for v in range(mdl.n_views_):
			selected_variables_.append((mdl.weights_[v] != 0)*1)
		self.selected_variables_ = selected_variables_

		self.canonical_correlations_indicies_ = corr_index
		self.train_canonical_correlations_ = train_canonical_correlation
		self.test_canonical_correlations_ = test_canonical_correlation
		self.n_components_ = n_components
		self.l1_sparsity_ = l1_sparsity
		self.model_obj_ = mdl
		return(self)

	def _bootstrap_correlation(self, x1, x2, n_bootstraps = 10000):
		assert x1.shape == x2.shape, "Error: x1 and x2 must have the same shape."
		assert x1.ndim <= 2, "Error: The maximum dimensions are two."
		n = len(x1)
		if x1.ndim == 2:
			corr_bootstraps = np.zeros((n_bootstraps, x1.shape[1]))
		else:
			x1 = x1[:,np.newaxis]
			x2 = x2[:,np.newaxis]
			corr_bootstraps = np.zeros((n_bootstraps, 1))
		for i in tqdm(range(n_bootstraps)):
			indices = np.random.choice(n, size=n, replace=True)
			x1_sample = x1[indices]
			x2_sample = x2[indices]
			for c in range(x1.shape[1]):
				corr_bootstraps[i,c] = pearsonr(x1_sample[:,c], x2_sample[:,c])[0]
		return(corr_bootstraps)

	def bootstrap_prediction_model(self, response_index, n_bootstraps = 10000):
		assert hasattr(self,'model_obj_'), "Error: run fit_model"
		# Training data
		y = self.train_scores_[response_index]
		y_hat = self.model_obj_.predict(self.train_scores_, response_index = response_index)
		prediction_train_r_ = np.zeros((self.n_components_))
		prediction_train_r_pval_ = np.zeros((self.n_components_))
		for c in range(self.n_components_):
			prediction_train_r_[c], prediction_train_r_pval_[c] = pearsonr(y[:, c], y_hat[:, c])
		self.prediction_model_train_y_ = y
		self.prediction_model_train_yhat_ = y_hat
		self.prediction_train_r_ = prediction_train_r_
		self.prediction_train_r_pval_ = prediction_train_r_pval_
		if n_bootstraps is not None:
			corr_bootstraps = self._bootstrap_correlation(y, y_hat, n_bootstraps = n_bootstraps)
			corr_bootstraps_pval = np.sum((corr_bootstraps < 0), 0) * 2 / n_bootstraps
			self.prediction_train_bootstraps_ = corr_bootstraps
			self.prediction_train_bootstraps_pvalue_ =  corr_bootstraps_pval
			self.prediction_train_bootstraps_CI_025_ = np.percentile(corr_bootstraps, 2.5, axis = 0)
			self.prediction_train_bootstraps_CI_975_ = np.percentile(corr_bootstraps, 97.5, axis = 0)
		# Test data
		y = self.test_scores_[response_index]
		y_hat = self.model_obj_.predict(self.test_scores_, response_index = response_index)
		prediction_test_r_ = np.zeros((self.n_components_))
		prediction_test_r_pval_ = np.zeros((self.n_components_))
		for c in range(self.n_components_):
			prediction_test_r_[c], prediction_test_r_pval_[c] = pearsonr(y[:, c], y_hat[:, c])
		self.prediction_model_test_y_ = y
		self.prediction_model_test_yhat_ = y_hat
		self.prediction_test_r_ = prediction_test_r_
		self.prediction_test_r_pval_ = prediction_test_r_pval_
		if n_bootstraps is not None:
			corr_bootstraps = self._bootstrap_correlation(y, y_hat, n_bootstraps = n_bootstraps)
			corr_bootstraps_pval = np.sum((corr_bootstraps < 0), 0) * 2 / n_bootstraps
			self.prediction_test_bootstraps_ = corr_bootstraps
			self.prediction_test_bootstraps_pvalue_ =  corr_bootstraps_pval
			self.prediction_test_bootstraps_CI_025_ = np.percentile(corr_bootstraps, 2.5, axis = 0)
			self.prediction_test_bootstraps_CI_975_ = np.percentile(corr_bootstraps, 97.5, axis = 0)

	# Candidate functions
	
	def bootstrap_model_loadings(self, n_bootstraps = 10000, bootstrap_training_loading = False):
		print("[Training Data]")
		lbs_pval_, lbs_CI_025_, lbs_CI_975_ = self.bootstrap_dataview_loadings(dataviews = self.views_train_, n_bootstraps = n_bootstraps)
		self.train_loadings_bootstrap_pval_ = lbs_pval_
		self.train_loadings_bootstrap_CI_025_ = lbs_CI_025_
		self.train_loadings_bootstrap_CI_975_ = lbs_CI_975_
		if bootstrap_training_loading:
			print("[Training Data]")
			lbs_pval_, lbs_CI_025_, lbs_CI_975_ = self.bootstrap_dataview_loadings(dataviews = self.views_test_, n_bootstraps = n_bootstraps)
			self.test_loadings_bootstrap_pval_ = lbs_pval_
			self.test_loadings_bootstrap_CI_025_ = lbs_CI_025_
			self.test_loadings_bootstrap_CI_975_ = lbs_CI_975_

	def _inner_correlation_func(self, b, mat, n_subs, seed):
		if seed is None:
			np.random.seed(np.random.randint(4294967295))
		elif seed == 'ignore':
			pass
		else:
			np.random.seed(seed)
		return(np.corrcoef(mat[choices(np.arange(0,mat.shape[0],1), k=n_subs)].T)[1:,0])
	def _outer_correlation_func(self, view_index, component_index, score, view, nonzeroweightidx, n_subs, n_bootstraps, seed):
		if seed is None:
			np.random.seed(np.random.randint(4294967295))
		else:
			np.random.seed(seed)
		mat = np.column_stack((score, view[:,nonzeroweightidx]))
		rs = np.corrcoef(mat.T)[1:,0]
		rs_dir = np.sign(rs)
		boot_rs = np.zeros((n_bootstraps, len(rs)))
		for b in range(n_bootstraps):
			boot_rs[b]  = self._inner_correlation_func(b = b, mat = mat, n_subs = n_subs, seed = 'ignore')
		boot_rs_pos = boot_rs*rs_dir
		rs_pval = np.ones((len(rs)))
		for e in range(len(rs)):
			temp = np.sort(boot_rs_pos[:,e])
			rs_pval[e] = np.divide(np.searchsorted(temp, 0), n_bootstraps)
		bs_loading_pval = np.ones((len(nonzeroweightidx)))
		bs_loading_025 = np.ones((len(nonzeroweightidx)))
		bs_loading_975 = np.ones((len(nonzeroweightidx)))
		bs_loading_pval[nonzeroweightidx] = rs_pval
		bs_loading_025[nonzeroweightidx] = np.percentile(boot_rs_pos, 2.5, axis = 0) * rs_dir
		bs_loading_025[~nonzeroweightidx] = np.nan
		bs_loading_975[nonzeroweightidx] = np.percentile(boot_rs_pos, 97.5, axis = 0) * rs_dir
		bs_loading_975[~nonzeroweightidx] = np.nan
		return(bs_loading_pval, bs_loading_025, bs_loading_975)
	def bootstrap_dataview_loadings(self, dataviews, n_bootstraps = 10000, parallel_use_inner = False):
		datascores = self.model_obj_.transform(dataviews)
		loadings_bootstrap_pval_ = []
		loadings_bootstrap_CI_025_ = []
		loadings_bootstrap_CI_975_ = []
		for view_index in tqdm(range(self.n_views_)):
			view = np.array(dataviews[view_index])
			n_subs, n_targets = view.shape
			bs_loading_pval = np.ones((self.model_obj_.weights_[view_index].shape))
			bs_loading_025 = np.ones((self.model_obj_.weights_[view_index].shape))
			bs_loading_975 = np.ones((self.model_obj_.weights_[view_index].shape))
			if parallel_use_inner:
				for component_index in range(self.n_components_):
					print("View[%d], Component[%d]: running %d iterations" % (view_index, component_index, n_bootstraps))
					score = np.array(datascores[view_index,:,component_index])
					nonzeroweightidx = self.model_obj_.weights_[view_index][:,component_index] != 0
					mat = np.column_stack((score, view[:,nonzeroweightidx]))
					rs = np.corrcoef(mat.T)[1:,0]
					rs_dir = np.sign(rs)
					boot_rs = np.zeros((n_bootstraps, len(rs)))
					seeds = generate_seeds(n_bootstraps)
					boot_rs = np.array(Parallel(n_jobs = self.n_jobs, backend='multiprocessing')(
								delayed(self._inner_correlation_func)(b = b, mat = mat, n_subs = n_subs, seed = seeds[b]) for b in range(n_bootstraps)))
					boot_rs_pos = boot_rs*rs_dir
					rs_pval = np.ones((len(rs)))
					for e in range(len(rs)):
						temp = np.sort(boot_rs_pos[:,e])
						rs_pval[e] = np.divide(np.searchsorted(temp, 0), n_bootstraps)
					bs_loading_pval[nonzeroweightidx, component_index] = rs_pval
					bs_loading_025[nonzeroweightidx, component_index] = np.percentile(boot_rs_pos, 2.5, axis = 0) * rs_dir
					bs_loading_025[~nonzeroweightidx, component_index] = np.nan
					bs_loading_975[nonzeroweightidx, component_index] = np.percentile(boot_rs_pos, 97.5, axis = 0) * rs_dir
					bs_loading_975[~nonzeroweightidx, component_index] = np.nan
			else:
				print("View[%d]: running %d iterations for %d components" % (view_index, n_bootstraps, self.n_components_))
				v = view_index
				seeds = generate_seeds(self.n_components_)
				output = Parallel(n_jobs = self.n_jobs, backend='multiprocessing')(
							delayed(self._outer_correlation_func)(view_index = v,
																	component_index = c,
																	score = np.array(datascores[v,:,c]),
																	view = view,
																	nonzeroweightidx = self.model_obj_.weights_[v][:,c]!=0,
																	n_subs = n_subs,
																	n_bootstraps = n_bootstraps, 
																	seed = seeds[c]) for c in range(self.n_components_))
				bs_loading_pval, bs_loading_025, bs_loading_975 = zip(*output)
				bs_loading_pval = np.array(bs_loading_pval).T
				bs_loading_025 = np.array(bs_loading_025).T
				bs_loading_975 = np.array(bs_loading_975).T
			loadings_bootstrap_pval_.append(bs_loading_pval)
			loadings_bootstrap_CI_025_.append(bs_loading_025)
			loadings_bootstrap_CI_975_.append(bs_loading_975)
		return(loadings_bootstrap_pval_, loadings_bootstrap_CI_025_, loadings_bootstrap_CI_975_)

# Plotting functions

def plot_ncomponents(model, views, max_n_comp, l1_sparsity, labels = None, png_basename = None):
	"""
	Plot the cumulative average variance explained (AVE) as a function of the number of components for a given SGCCA model.

	Parameters:
	--------
		model : object
			A SGCCA object that contains a design matrix and scheme.
		views : list
			A list of views (numpy arrays) used to train the SGCCA model.
		max_n_comp: int or ndarray
			The maximum number of components to plot the AVE. It can be an array with each view having a value.
		l1_sparsity: float  or np.ndarray
			The sparsity parameter for each view. If a scalar float, it is used for all views. If a numpy array, it must have shape (max_n_comp, n_views).
		labels (list, optional): A list of strings to label each view. Default is None, which labels each view as 'View i'.
		png_basename (str, optional): The base name for the output PNG file. If None, the plot is displayed instead of being saved. Default is None.

	Returns:
	--------
		None
	"""

	assert hasattr(model,'train_index_'), "Error: no training data index. Run create_nfold."
	views_train_ = model.subsetviews(views, model.train_index_)
	n_views_ = len(views_train_)
	if np.isscalar(l1_sparsity):
		l1_sparsity = np.repeat(l1_sparsity, n_views_)
	if l1_sparsity.shape != (np.max(max_n_comp), n_views_):
		l1_sparsity = np.tile(l1_sparsity, np.max(max_n_comp)).reshape(np.max(max_n_comp), n_views_)
	temp_model = sgcca_rwrapper(design_matrix = model.design_matrix, l1_sparsity = l1_sparsity, n_comp = max_n_comp, scheme = model.scheme, tol=1e-8).fit(views_train_)
	x_values = np.arange(0, np.max(max_n_comp), 1)
	if labels is None:
		labels = ["View %d" % (i+1) for i in range(temp_model.n_views_)]
	fig, ax = plt.subplots(figsize=(8, 6))
	sns.set(style="ticks", font_scale=1.3)
	sns.lineplot(x=x_values, y=cumulative_sum_of_array(temp_model.AVE_outer_), linewidth=2, color='k', label='Model')
	colors = sns.color_palette("Set1", n_colors=len(labels))
	for i, label in enumerate(labels):
		sns.lineplot(x=x_values, y=cumulative_sum_of_array(temp_model.AVE_views_[i]), linewidth=2, color=colors[i], label=label, ls='--')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.tick_params(axis='both', which='both')
	plt.xlabel('Number of Components')
	plt.ylabel('Cumulative AVE')
	plt.xlim([1, np.max(max_n_comp)])
	plt.ylim([0., 1.])
	ax.legend(loc="best", fontsize=10)
	sns.despine()
	plt.tight_layout()
	if png_basename is not None:
		plt.savefig("%s_plot_ncomponentspng" % png_basename)
		plt.close()
	else:
		plt.show()

def plot_parameter_selection(model, xlabel = "Sparsity", ylabel = "Tuning metric (scaled)", L1_penalty_range = np.arange(0.1,1.1,.1), scale_tuning_metric = True, png_basename = None):
	tmetric = np.array(model.parameterselection_tmetric_)
	tstar_ = np.array(model.parameterselection_tstar_)
	if scale_tuning_metric:
		for l, pen in enumerate(L1_penalty_range):
			tmetric[l] = tmetric[l] - np.mean(tstar_[l])
			tmetric[l] = tmetric[l] / np.std(tstar_[l])
			tstar_[l] = tstar_[l] - np.mean(tstar_[l])
			tstar_[l] = tstar_[l] / np.std(tstar_[l])
	plt.subplot(211)
	plt.plot(L1_penalty_range, model.parameterselection_zstat_, 'ko--', linewidth=1)
	plt.ylabel("Z-statistic")
	x1,x2,y1,y2 = plt.axis()
	x2 = 1.1
	y2 = np.round(y2) + 1
	plt.xlim(0, x2)
	plt.ylim(y1, y2)
	plt.xticks(L1_penalty_range)
	plt.subplot(212)
	plt.plot(L1_penalty_range, tmetric, 'ko', linewidth=1)
	for l, pen in enumerate(L1_penalty_range):
		jitter = pen + np.random.normal(0, scale = L1_penalty_range[0]*0.05, size=len(tstar_[0]))
		permz = tstar_[l]
		plt.scatter(jitter, permz, c = 'b', marker='.', alpha = 0.5)
	x1,x2,y1,y2 = plt.axis()
	x2 = 1.1
	y2 = np.round(y2) + 1
	plt.xlim(0, x2)
#	plt.ylim(0, y2)
	plt.xticks(L1_penalty_range)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.tight_layout()
	if png_basename is not None:
		plt.savefig("%s_parameter_selection.png" % png_basename)
		plt.close()
	else:
		plt.show()

def scatter_histogram(x, y, xlabel = None, ylabel = None, png_basename = None):
	"""
	Scatter plot best fit line as well as with histograms with gaussian_kde curves
	
	e.g.,
	scatter_hist(yhat, Y_, 'Neuroimaging Variates', 'Clinical Variates')
	"""
	def _scatter_hist(x, y, ax, ax_histx, ax_histy, xlabel = None, ylabel = None):
		ax_histx.tick_params(axis="x", labelbottom=False)
		ax_histy.tick_params(axis="y", labelleft=False)
		sns.regplot(x = x, y = y, ax = ax)
		x0,x1 = ax.get_xlim()
		y0,y1 = ax.get_ylim()
		if len(np.arange(x0, x1, x.var()*4)) > 30:
			xbins = 30
		else:
			xbins = np.arange(x0, x1, x.var()*4)
		ax_histx.hist(x, bins=xbins, density=True)
		xs_ = np.linspace(x0, x1, 301)
		kde = gaussian_kde(x)
		ax_histx.plot(xs_, kde.pdf(xs_))
		ax_histy.hist(y, bins=np.arange(y0, y1, y.var()*2), orientation='horizontal', density=True)
		ys_ = np.linspace(y0, y1, 301)
		kde = gaussian_kde(y)
		ax_histy.plot(kde.pdf(ys_), ys_)
	fig = plt.figure(figsize=(8, 6))
	# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
	# the size of the marginal axes and the main axes in both directions.
	# Also adjust the subplot parameters for a square plot.
	gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
								left=0.2, right=0.9, bottom=0.1, top=0.9,
								wspace=0.05, hspace=0.05)
	# Create the Axes.
	ax = fig.add_subplot(gs[1, 0])
	if xlabel is not None:
		ax.set_xlabel(xlabel)
	if ylabel is not None:
		ax.set_ylabel(ylabel)
	ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
	ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
	# Draw the scatter plot and marginals.
	_scatter_hist(x, y, ax, ax_histx, ax_histy, xlabel = xlabel, ylabel = ylabel)
	if png_basename is not None:
		plt.savefig("%s_scatter_histogram.png" % png_basename)
		plt.close()
	else:
		plt.show()

def plot_prediction_bootstraps(model, png_basename = None):
	component_indices = np.arange(model.n_components_)
	# Plot the data as a grouped errorbar plot
	fig, ax = plt.subplots()
	ax.scatter(model.prediction_train_bootstraps_.mean(0), component_indices - 0.1, marker='s', label='Training')
	ax.hlines(component_indices - 0.1, model.prediction_train_bootstraps_CI_025_, model.prediction_train_bootstraps_CI_975_, color='#1f77b4')
	ax.scatter(model.prediction_test_bootstraps_.mean(0), component_indices + 0.1, marker='s', label='Test')
	ax.hlines(component_indices + 0.1, model.prediction_test_bootstraps_CI_025_, model.prediction_test_bootstraps_CI_975_, color='#ff7f0e')
	ax.set_yticks(component_indices)
	ax.set_yticklabels([f'Component {i+1}' for i in component_indices])
	ax.set_ylabel('SGCCA Component')
	ax.set_xlabel('Prediction (r)')
	ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
	ax.invert_yaxis()
	plt.axvline(0, ls = '--', color = 'k')
	plt.tight_layout()
	if png_basename is not None:
		plt.savefig("%s_bootstrap_prediction.png" % png_basename)
		plt.close()
	else:
		plt.show()

