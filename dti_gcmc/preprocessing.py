from __future__ import division
from __future__ import print_function

import os
import h5py
import operator
import numpy as np
import pandas as pd
import cPickle as pkl
import scipy.sparse as sp

from collections import Counter
from data_utils import load_data, map_data
from utils import user_based_probability_distribution
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import TruncatedSVD

def normalize_features(feat):

	degree = np.asarray(feat.sum(1)).flatten()

	# set zeros to inf to avoid dividing by zero
	degree[degree == 0.] = np.inf

	degree_inv = 1. / degree
	degree_inv_mat = sp.diags([degree_inv], [0])
	feat_norm = degree_inv_mat.dot(feat)

	if feat_norm.nnz == 0:
		print('ERROR: normalized adjacency matrix has only zero entries!!!!!')
		exit

	return feat_norm


def load_matlab_file(path_file, name_field):
	"""
	load '.mat' files
	inputs:
		path_file, string containing the file path
		name_field, string containig the field name (default='shape')
	warning:
		'.mat' files should be saved in the '-v7.3' format
	"""
	db = h5py.File(path_file, 'r')
	ds = db[name_field]
	try:
		if 'ir' in ds.keys():
			data = np.asarray(ds['data'])
			ir = np.asarray(ds['ir'])
			jc = np.asarray(ds['jc'])
			out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
	except AttributeError:
		# Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
		out = np.asarray(ds).astype(np.float32).T

	db.close()

	return out


def preprocess_user_item_features(u_features, v_features):
	"""
	Creates one big feature matrix out of user features and item features.
	Stacks item features under the user features.
	"""
	zero_csr_u = sp.csr_matrix((u_features.shape[0], v_features.shape[1]), dtype=u_features.dtype)
	zero_csr_v = sp.csr_matrix((v_features.shape[0], u_features.shape[1]), dtype=v_features.dtype)

	u_features = sp.hstack([u_features, zero_csr_u], format='csr')
	v_features = sp.hstack([zero_csr_v, v_features], format='csr')

	return u_features, v_features


def globally_normalize_bipartite_adjacency(adjacencies, verbose=False, symmetric=True):
	""" Globally Normalizes set of bipartite adjacency matrices """

	if verbose:
		print('Symmetrically normalizing bipartite adj')
	# degree_u and degree_v are row and column sums of adj+I

	adj_tot = np.sum(adj for adj in adjacencies)
	degree_u = np.asarray(adj_tot.sum(1)).flatten()
	degree_v = np.asarray(adj_tot.sum(0)).flatten()

	# set zeros to inf to avoid dividing by zero
	degree_u[degree_u == 0.] = np.inf
	degree_v[degree_v == 0.] = np.inf

	degree_u_inv_sqrt = 1. / np.sqrt(degree_u)
	degree_v_inv_sqrt = 1. / np.sqrt(degree_v)
	degree_u_inv_sqrt_mat = sp.diags([degree_u_inv_sqrt], [0])
	degree_v_inv_sqrt_mat = sp.diags([degree_v_inv_sqrt], [0])

	degree_u_inv = degree_u_inv_sqrt_mat.dot(degree_u_inv_sqrt_mat)

	if symmetric:
		adj_norm = [degree_u_inv_sqrt_mat.dot(adj).dot(degree_v_inv_sqrt_mat) for adj in adjacencies]

	else:
		adj_norm = [degree_u_inv.dot(adj) for adj in adjacencies]

	return adj_norm


def sparse_to_tuple(sparse_mx):
	""" change of format for sparse matrix. This format is used
	for the feed_dict where sparse matrices need to be linked to placeholders
	representing sparse matrices. """

	if type(sparse_mx) == tuple:
		return sparse_mx[0], sparse_mx[1], sparse_mx[2]

	if not sp.isspmatrix_coo(sparse_mx):
		sparse_mx = sparse_mx.tocoo()
	coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
	values = sparse_mx.data
	shape = sparse_mx.shape
	return coords, values, shape

def create_trainvaltest_split(dataset, seed=1234, testing=False, datasplit_path=None, datasplit_from_file=False,
							  split_type='stratified', sampling_scheme='simple_stratified', neg_sample_size=30000, 
							  verbose=True, model='gcmc'):
	"""
	Splits data set into train/val/test sets from full bipartite adjacency matrix. Shuffling of dataset is done in
	load_data function.
	For each split computes 1-of-num_classes labels. Also computes training
	adjacency matrix.
	"""

	sampling_scheme = 'simple_stratified'

	if split_type == 'random':
		u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
			val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values, \
			class_weights = random_sampling(dataset, model, seed, 
											testing, datasplit_path, 
											datasplit_from_file, model, 
											verbose)

	elif split_type == 'stratified':
		u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
			val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values, \
			class_weights = stratified_sampling(dataset, model, sampling_scheme, 
												neg_sample_size, seed, 
												testing, datasplit_path, 
												datasplit_from_file,
												verbose)

	else:
		raise NotImplementedError('Sampling type not supported yet.')

	return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
		val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values, \
		class_weights

def random_sampling(dataset, model, seed=1234, testing=False, 
				datasplit_path=None, datasplit_from_file=False, 
				verbose=True):
	
	""" Splits the dataset into train, val and test by randomly sampling
		from the dataset. """

	print ('Using random split...')
	if datasplit_from_file and os.path.isfile(datasplit_path):
		print('Reading dataset splits from file...')
		with open(datasplit_path) as f:
			num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features = pkl.load(f)

		if verbose:
			print('Number of users = %d' % num_users)
			print('Number of items = %d' % num_items)
			print('Number of links = %d' % ratings.shape[0])
			print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))

	else:
		num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features = load_data(dataset, seed=seed,
																							verbose=verbose)

		with open(datasplit_path, 'w') as f:
			pkl.dump([num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features], f)

	neutral_rating = -1

	rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

	labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
	labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])
	labels = labels.reshape([-1])

	# number of test and validation edges
	num_test = int(np.ceil(ratings.shape[0] * 0.2))
	if dataset == 'ml_100k':
		num_val = int(np.ceil(ratings.shape[0] * 0.8 * 0.1))
	else:
		num_val = int(np.ceil(ratings.shape[0] * 0.8 * 0.1))

	num_train = ratings.shape[0] - num_val - num_test

	pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])

	idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

	train_idx = idx_nonzero[0:num_train]
	val_idx = idx_nonzero[num_train:num_train + num_val]
	test_idx = idx_nonzero[num_train + num_val:]

	train_pairs_idx = pairs_nonzero[0:num_train]
	val_pairs_idx = pairs_nonzero[num_train:num_train + num_val]
	test_pairs_idx = pairs_nonzero[num_train + num_val:]

	u_test_idx, v_test_idx = test_pairs_idx.transpose()
	u_val_idx, v_val_idx = val_pairs_idx.transpose()
	u_train_idx, v_train_idx = train_pairs_idx.transpose()

	# create labels
	train_labels = labels[train_idx]
	val_labels = labels[val_idx]
	test_labels = labels[test_idx]

	if testing:
		u_train_idx = np.hstack([u_train_idx, u_val_idx])
		v_train_idx = np.hstack([v_train_idx, v_val_idx])
		train_labels = np.hstack([train_labels, val_labels])
		# for adjacency matrix construction
		train_idx = np.hstack([train_idx, val_idx])

	# make training adjacency matrix
	rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
	rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
	rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

	class_values = np.sort(np.unique(ratings))
	class_weights = compute_class_weight("balanced", class_values, np.concatenate([train_labels, 
		test_labels], axis=0))

	return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
		val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values, \
		class_weights

def stratified_sampling(dataset, model, sampling_scheme, neg_sample_size, 
				seed=1234, testing=False, datasplit_path=None, 
				datasplit_from_file=False, verbose=True):

	""" Splits the dataset into train, val and test by drawing
		stratified samples from it.

		:params sampling_scheme: Defines the scheme used for
			sampling. Available types are:
			['simple_stratified', 'oversample', 'user_based', 'item_based'] 

		Taken from Pan et al. (2008). """

	sampling_scheme = 'oversample'

	np.random.seed(seed)

	if datasplit_from_file and os.path.isfile(datasplit_path):
		print('Reading dataset splits from file...')
		with open(datasplit_path) as f:
			num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features = pkl.load(f)

		if verbose:
			print('Number of users = %d' % num_users)
			print('Number of items = %d' % num_items)
			print('Number of links = %d' % ratings.shape[0])
			print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))

	else:
		num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features = load_data(dataset, seed=seed,
																							verbose=verbose)

	print('Reducing dimension using SVD...')

	u_features = u_features.todense()
	v_features = v_features.todense()

	u_svd = TruncatedSVD(n_components=100, n_iter=5)
	v_svd = TruncatedSVD(n_components=100, n_iter=5)

	u_features = u_svd.fit_transform(u_features)
	v_features = v_svd.fit_transform(v_features)

	u_features = sp.csr_matrix(u_features)
	v_features = sp.csr_matrix(v_features)

	neutral_rating = -1

	rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

	labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
	labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])
	labels = labels.reshape([-1])

	graph = [u_nodes, v_nodes, ratings]
	graph = np.vstack(graph)
	graph = graph.T

	# Sort based on ratings and split for 0 and 1 ratings
	graph = graph[graph[:, 2].argsort()]
	graph_0, graph_1 = np.split(graph, np.where(np.diff(graph[:,2]))[0]+1)

	np.random.shuffle(graph_0)
	np.random.shuffle(graph_1)

	# Create train-val-test split

	graph_0_train = graph_0[:int(0.7 * graph_0.shape[0])]
	graph_0_val = graph_0[int(0.7 * graph_0.shape[0]):int(0.8 * graph_0.shape[0])]
	graph_0_test = graph_0[int(0.8 * graph_0.shape[0]):]

	graph_1_train = graph_1[:int(0.7 * graph_1.shape[0])]
	graph_1_val = graph_1[int(0.7 * graph_1.shape[0]):int(0.8 * graph_1.shape[0])]
	graph_1_test = graph_1[int(0.8 * graph_1.shape[0]):]

	# # Create train-val split
	# graph_val = graph_train[:int(graph_train.shape[0]*0.1)]
	# graph_train = graph_train[int(graph_train.shape[0]*0.1):]

	if graph_0_train.shape[0] < neg_sample_size:
		err = 'Size of negative sample is higher than no. of negative samples in train set.'
		raise ValueError(err)

	if sampling_scheme == 'simple_stratified':
		print('Using simple stratified split...')

	elif sampling_scheme == 'oversample':
		print('Using stratified split with oversampling...')
		index = np.random.choice(graph_0_train.shape[0], neg_sample_size, replace=False)
		graph_0_train = graph_0_train[index]
		index = np.random.choice(graph_1_train.shape[0], neg_sample_size, replace=True)
		graph_1_train = graph_1_train[index]

	elif sampling_scheme == 'user_based':
		print('Using user-based stratified split...')

		prob_weights, items = user_based_probability_distribution(u_nodes,
														   v_nodes, 
														   graph_0_train, 
														   graph_1_train)

		print (graph_0_train.shape, len(prob_weights))
		index = np.random.choice(graph_0_train.shape[0], neg_sample_size, p=prob_weights, 
				replace=False)
		graph_0_train = graph_0_train[index]

	elif sampling_scheme == 'item_based':
		raise NotImplementedError('Sampling scheme not supported yet.')

	else:
		raise ValueError('Sampling scheme not recognized.')

	graph_train = np.concatenate([graph_0_train, graph_1_train], axis=0)
	graph_test = np.concatenate([graph_0_test, graph_1_test], axis=0)
	graph_val = np.concatenate([graph_0_val, graph_1_val], axis=0)

	np.random.shuffle(graph_train)
	np.random.shuffle(graph_test)
	np.random.shuffle(graph_val)

	train_idx = np.array([u * num_items + v for u, v, _ in graph_train])
	val_idx = np.array([u * num_items + v for u, v, _ in graph_val])

	u_train_idx, v_train_idx, train_labels = graph_train.T
	u_test_idx, v_test_idx, test_labels = graph_test.T
	u_val_idx, v_val_idx, val_labels = graph_val.T

	if testing:
		u_train_idx = np.hstack([u_train_idx, u_val_idx])
		v_train_idx = np.hstack([v_train_idx, v_val_idx])
		train_labels = np.hstack([train_labels, val_labels])
		# for adjacency matrix construction
		train_idx = np.hstack([train_idx, val_idx])
		
	# make training adjacency matrix
	rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
	rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
	rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

	class_values = np.sort(np.unique(ratings))
	class_weights = compute_class_weight("balanced", class_values, np.concatenate([train_labels, 
		test_labels], axis=0))

	return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
	val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values, \
	class_weights