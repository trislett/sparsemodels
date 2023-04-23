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

from scipy.stats import t as tdist
from scipy.stats import f as fdist
from scipy.stats import norm, chi2, pearsonr, gaussian_kde
from scipy.linalg import pinv
from statsmodels.stats.multitest import multipletests, fdrcorrection
from joblib import Parallel, delayed

from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.cross_decomposition import PLSRegression, CCA

from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
# suppress console because of weird permission around r
from rpy2.rinterface import RRuntimeWarning

warnings.filterwarnings("ignore", category=RRuntimeWarning)
warnings.filterwarnings('ignore') 

from sparsemodels.cynumstats import cy_lin_lstsqr_mat_residual, cy_lin_lstsqr_mat, fast_se_of_slope

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

# Sparse Generalized Canonical Correlation Analysis for Multiblock Data

class sgcca_rwrapper:
	"""
	Wrapper class for the SGCCA function of the R package RGCCA.
	https://rdrr.io/cran/RGCCA/man/sgcca.html
	"""
	def __init__(self, design_matrix = None, l1_sparsity = None, n_comp = 1, scheme = "centroid", scale = True, init = "svd", bias = True, effective_zero = sys.float_info.epsilon):
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
			Schheme options are "horst", "factorial" or "centroid" (Default: "centroid")
		scale : bool
			A boolean that specifies whether to scale the views before running SGCCA.
			Default value is True.
		init : str
			A string that specifies the initialization method used to initialize the optimization problem.
			Default value is "svd".
		bias : bool
			A boolean that specifies whether to include a bias term in the optimization problem.
			Default value is True.
		effective_zero : float
			A float that specifies a small value to use as an effective zero.

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
		self.effective_zero = effective_zero

	def scaleviews(self, views, centre = True, scale = True, div_sqr_numvar = True, axis = 0):
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
		div_sqr_numvar : bool
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
			if div_sqr_numvar:
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

	def check_sparsity(self):
		"""
		Checks if l1_sparsity is valid and adjusts it if necessary.
		"""
		for v in range(self.n_views_):
			sthrehold = 1 / np.sqrt(self.views_[v].shape[1])
			sparsity = self.l1_sparsity[v]
			if sparsity < sthrehold:
				nsparsity = np.round(np.round(sthrehold, 4) + 0.0001, 4)
				print("Sparsity of view[%d] is too low. Adjusting to new value = %1.4f" %(int(v), nsparsity))
				self.l1_sparsity[:,v] = nsparsity

	def fit(self, X):
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
		# l1 contraints to the outer weights ranging from 1/sqrt(view[j].shape[1]) to 1. i.e., 1 / sqrt(nvars) per data view is the minimum sparsity.
		if self.l1_sparsity is None:
			self.l1_sparsity = np.repeat(1., self.n_views_)
		if np.isscalar(self.l1_sparsity):
			self.l1_sparsity = np.repeat(self.l1_sparsity, self.n_views_)
		if np.isscalar(self.n_comp):
			self.n_comp = np.repeat(self.n_comp, self.n_views_)
		if self.scale:
			self.views_ = self.scaleviews(self.views_)
		self.check_sparsity()
		numpy2ri.activate()
		fit = rgcca.sgcca(A = self.views_, 
							C = self.design_matrix,
							c1 = np.tile(self.l1_sparsity, np.max(self.n_comp)).reshape(np.max(self.n_comp), self.n_views_),
							ncomp = self.n_comp, 
							scheme = self.scheme,
							scale = False,
							init = self.init,
							bias = self.bias,
							tol = self.effective_zero,
							verbose  = False)
		numpy2ri.deactivate()

		self.scores_ = np.array(fit.rx2('Y'))
		self.weights_outer_ = self._rlist_to_nplist(fit.rx2('a'))
		self.weights_ = self._rlist_to_nplist(fit.rx2('astar'))
		self.AVE_views_ = np.array(fit.rx2('AVE')[0]) # this is the mean of the structural coefficents
		self.AVE_outer_ = np.array(fit.rx2('AVE')[1])
		self.AVE_inner_ = np.array(fit.rx2('AVE')[2])
		return(self)

	def transform(self, views, calculate_loadings = False, outer = False):
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
		X = self.scaleviews((list([view]))[0])
		scores = []
		for v in range(self.n_views_):
			scores.append(np.dot(views[X], self.weights_[v]))
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
		"""
		if scores.ndim == 2:
			scores = scores[:,:,np.newaxis]
		n_view, n_subjects, n_comps = scores.shape
		independent_index = np.arange(0, n_view, 1)
		independent_index = independent_index[independent_index!=response_index]
		yhat = np.zeros((scores[response_index,:,:].shape))
		for i in range(10):
			X_ = scores[independent_index,:,i].T
			Y_ = scores[response_index,:,i].T
			reg = LinearRegression(fit_intercept=False).fit(X_,Y_)
			if verbose:
				R2score = reg.score(X_,Y_)
				print("Component [%d] R2(score) = %1.3f" % (int(i+1), R2score))
			yhat[:, i] = reg.predict(X_)
		return(yhat)

class parallel_sgcca():
	def __init__(self, n_jobs = 8, n_permutations = 10000):
		"""
		Main SGCCA function
		"""
		self.n_jobs = n_jobs
		self.n_permutations = n_permutations
	def _datestamp(self):
		print("2023_21_04")
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
			the index array for each fold (n_folds, training_fold_size)
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
	def create_nfold(self, group, n_fold = 10, holdout = 0.3, verbose = True):
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
		subsetdata = []
		for v in range(len(views)):
			subsetdata.append(views[v][indices])
		return(subsetdata)
	def permute_views(self, views, seed = None):
		"""
		Randomly permutes each data view
		"""
		if seed is None:
			np.random.seed(np.random.randint(4294967295))
		else:
			np.random.seed(seed)
		permutedviews = []
		for v in range(len(views)):
			permutedviews.append(np.random.permutation(views[v]))
		return(permutedviews)
	def prediction_cv(self, l1_range, n_perm_per_block = 100):
		"""
		Montecarlo Cross-validation gridseach
		"""
		pass
	def _premute_model(self, p, views, metric, view_index):
		"""
		"""
		pass
	def run_parallel_parameterselection(self, metric = 'mean_correlation', view_index = None, L1_penalty_range = np.arange(0.1,1.1,.1), nperms = 100):
		"""
		Parameters
		----------
		metric: str
			Metric options are: fisherz_transformation, prediction, or mean_correlation. (Default: mean_correlation.)
		view_index: None or int
			Sets the view to optimize. Must be set of for prediction. If None, all pairwise correlations are used.
		Returns
		---------
			self
		"""
		assert hasattr(self,'train_index_'), "Error: run create_nfold"
		
		views_train = self.subsetviews(views, self.train_index_)
		n_views = len(views_train)
		self.nviews_ = n_views
# Plotting functions

def scatter_histogram(x, y, xlabel = None, ylabel = None):
	"""
	Scatter plot best fit line as well as with histograms with gaussian_kde curves
	
	e.g.,
	scatter_hist(yhat, Y_, 'Neuroimaging Variates', 'Clinical Variates')
	plt.tight_layout()
	plt.show()
	"""
	def _scatter_hist(x, y, ax, ax_histx, ax_histy, xlabel = None, ylabel = None):
		ax_histx.tick_params(axis="x", labelbottom=False)
		ax_histy.tick_params(axis="y", labelleft=False)
		sns.regplot(x = x, y = y, ax = ax)
		x0,x1 = ax.get_xlim()
		y0,y1 = ax.get_ylim()
		binwidth = 0.01
		ax_histx.hist(x, bins=np.arange(x0, x1, x.var()*2), density=True)
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
								left=0.1, right=0.9, bottom=0.1, top=0.9,
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







