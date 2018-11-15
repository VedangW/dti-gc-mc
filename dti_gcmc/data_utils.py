from __future__ import division
from __future__ import print_function

import random
import pickle

import numpy as np
import pandas as pd

import scipy.sparse as sp

def data_iterator(data, batch_size):
	"""
	A simple data iterator from https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/
	:param data: list of numpy tensors that need to be randomly batched across their first dimension.
	:param batch_size: int, batch_size of data_iterator.
	Assumes same first dimension size of all numpy tensors.
	:return: iterator over batches of numpy tensors
	"""
	# shuffle labels and features
	max_idx = len(data[0])
	idxs = np.arange(0, max_idx)
	np.random.shuffle(idxs)
	shuf_data = [dat[idxs] for dat in data]

	# Does not yield last remainder of size less than batch_size
	for i in range(max_idx//batch_size):
		data_batch = [dat[i*batch_size:(i+1)*batch_size] for dat in shuf_data]
		yield data_batch


def map_data(data):
	"""
	Map data to proper indices in case they are not in a continues [0, N) range

	Parameters
	----------
	data : np.int32 arrays

	Returns
	-------
	mapped_data : np.int32 arrays
	n : length of mapped_data

	"""
	uniq = list(set(data))

	id_dict = {old: new for new, old in enumerate(sorted(uniq))}
	data = np.array(map(lambda x: id_dict[x], data))
	n = len(uniq)

	return data, id_dict, n


def load_data(fname, seed=1234, verbose=True):
	""" Loads dataset and creates adjacency matrix
	and feature matrix

	Parameters
	----------
	fname : str, dataset
	seed: int, dataset shuffling seed
	verbose: to print out statements or not

	Returns
	-------
	num_users : int
		Number of users and items respectively

	num_items : int

	u_nodes : np.int32 arrays
		User indices

	v_nodes : np.int32 array
		item (movie) indices

	ratings : np.float32 array
		User/item ratings s.t. ratings[k] is the rating given by user u_nodes[k] to
		item v_nodes[k]. Note that that the all pairs u_nodes[k]/v_nodes[k] are unique, but
		not necessarily all u_nodes[k] or all v_nodes[k] separately.

	u_features: np.float32 array, or None
		If present in dataset, contains the features of the users.

	v_features: np.float32 array, or None
		If present in dataset, contains the features of the users.

	seed: int,
		For datashuffling seed with pythons own random.shuffle, as in CF-NADE.

	"""

	u_features = None
	v_features = None

	print('Loading dataset', fname)

	data_dir = 'data/' + fname


	if fname == 'dataset_1':
		store_repo = 'data/dti_store/'

		with open(store_repo + 'graph_1.pkl') as f:
			graph_info = pickle.load(f)

		num_users, num_items, u_nodes_ratings, \
		v_nodes_ratings, ratings, u_features, \
		v_features = graph_info

	elif fname == 'dataset_2':
		store_repo = 'data/dti_store/'

		with open(store_repo + 'graph_2.pkl') as f:
			graph_info = pickle.load(f)

		num_users, num_items, u_nodes_ratings, \
		v_nodes_ratings, ratings, u_features, \
		v_features = graph_info

	else:
		raise ValueError('Dataset name not recognized: ' + fname)

	if verbose:
		print('Number of users = %d' % num_users)
		print('Number of items = %d' % num_items)
		print('Number of links = %d' % ratings.shape[0])
		print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))

	return num_users, num_items, u_nodes_ratings, v_nodes_ratings, ratings, u_features, v_features