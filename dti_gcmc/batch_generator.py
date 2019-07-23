from __future__ import print_function

import numpy as np
import scipy.sparse as sp

class BatchGenerator():

	def __init__(self, testing, val_size=0.15, test_size=0.2, 
					neg_sample_size=100000, verbose=True):
		""" Builds a sample for each epoch for training
			in the GCMC algorithm. 
			
			:param neg_sample_size: no. of negative samples to
				choose from all negative samples for batch.
			:param test_size: fraction of full dataset to use
				as test set.
			:param val_size: fraction of train set to use as the
				validation set. """

		self.call = False
		self.testing = testing
		self.verbose = verbose

		self.val_size = val_size
		self.test_size = test_size
		self.train_size = (1. - self.test_size)*(1. - self.val_size)

		self.neg_sample_size = neg_sample_size
		

	def __call__(self, graph_info):
		""" Fit the BatchGenerator object on the dataset. 

			:params graph_info: Specifics of the dataset.
				graph_info = [num_users, num_items, u_nodes,
				v_nodes, ratings, u_features, v_features]. 
			:returns Specifics of the test set, features
				and class values. """

		self.graph_info = graph_info

		num_users, num_items, u_nodes, v_nodes, \
		ratings, u_features, v_features = self.graph_info

		self.num_users = num_users
		self.num_items = num_items
		self.u_nodes = u_nodes
		self.v_nodes = v_nodes
		self.ratings = ratings
		self.u_features = u_features
		self.v_features = v_features

		self.test_set_size = ratings.shape[0] * self.test_size
		self.val_set_size = ratings.shape[0] * \
							(1. - self.test_size) * self.val_size
		self.train_set_size = ratings.shape[0] - \
							(self.test_set_size + self.val_set_size)

		if self.verbose:
			print('Number of users = %d' % num_users)
			print('Number of items = %d' % num_items)
			print('Number of links = %d' % ratings.shape[0])
			print('Fraction of positive links = %.4f' % \
				(float(ratings.shape[0]) / (num_users * num_items)))

			print ('')
			print ('Size of train set = %d samples' % self.train_set_size)
			print ('Size of val set = %d samples' % self.val_set_size)
			print ('Size of test set = %d samples' % self.test_set_size)

		graph = [self.u_nodes, self.v_nodes, self.ratings]
		self.graph = np.vstack(graph)

		neutral_rating = -1

		# Map each class to a rating
		self.rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

		# Create the list of labels
		labels = np.full((self.num_users, self.num_items),
						 neutral_rating, dtype=np.int32)
		labels[u_nodes, v_nodes] = np.array([self.rating_dict[r] for r in self.ratings])
		self.labels = labels.reshape([-1])

		graph = self.graph.T

		# Sort based on ratings and split for 0 and 1 ratings
		graph = graph[graph[:, 2].argsort()]
		graph_0, graph_1 = np.split(graph, np.where(np.diff(graph[:,2]))[0]+1)

		# Randomly shuffle pos and neg samples 
		np.random.shuffle(graph_0)
		np.random.shuffle(graph_1)

		# Create train-test split
		train_split = 1 - self.test_size

		# Split negative samples into train and test
		graph_0_train = graph_0[:int(train_split * graph_0.shape[0])]
		graph_0_test = graph_0[int(train_split * graph_0.shape[0]):]

		# Split positive samples into train and test
		graph_1_train = graph_1[:int(train_split * graph_1.shape[0])]
		graph_1_test = graph_1[int(train_split * graph_1.shape[0]):]

		# Create train and test sets by merging splits
		self.graph_train = np.concatenate([graph_0_train, 
									graph_1_train], 
									axis=0)
		self.graph_test = np.concatenate([graph_0_test, 
									graph_1_test], 
									axis=0)

		# Randomly shuffle train and test
		np.random.shuffle(self.graph_train)
		np.random.shuffle(self.graph_test)

		u_test_idx, v_test_idx, test_labels = self.graph_test.T

		# All classes
		self.class_values = np.sort(np.unique(self.ratings))

		# Object has been called
		self.call = True

		return self.u_features, self.v_features, \
				test_labels, u_test_idx, v_test_idx, \
				self.class_values

	def next(self):
		""" Generate next batch from the train set.
			Only to be called after using __call__(). 

			next() is not the same as __next__() as
			generating a list of train and val sets
			at once is extremely memory inefficient. """

		# Check if dataset has been fit yet.
		if not self.call:
			err = "BatchGenerator needs to be fit on dataset."
			raise RuntimeError(err)

		graph_train = self.graph_train

		# Sort based on ratings and split for 0 and 1 ratings
		graph_train = graph_train[graph_train[:, 2].argsort()]
		graph_0_train, graph_1_train = np.split(graph_train, 
				np.where(np.diff(graph_train[:,2]))[0]+1)

		# Randomly select only neg_sample_size samples
		# from negative training samples.
		try:
			index = np.random.choice(graph_0_train.shape[0], 
									self.neg_sample_size, 
									replace=False)
			graph_0_train = graph_0_train[index]
		except ValueError as e:
			err = "neg_sample_size is more than size of training set."
			raise ValueError(err)

		train_split = 1 - self.val_size

		# Split negative samples into train and test
		graph_0_train = graph_0_train[:int(train_split * graph_0_train.shape[0])]
		graph_0_val = graph_0_train[int(train_split * graph_0_train.shape[0]):]

		# Split positive samples into train and test
		graph_1_train = graph_1_train[:int(train_split * graph_1_train.shape[0])]
		graph_1_val = graph_1_train[int(train_split * graph_1_train.shape[0]):]

		# Create train and test sets by merging splits
		graph_train = np.concatenate([graph_0_train, 
									graph_1_train], 
									axis=0)
		graph_val = np.concatenate([graph_0_val, 
									graph_1_val], 
									axis=0)

		# Randomly shuffle
		np.random.shuffle(graph_train)
		np.random.shuffle(graph_val)

		# Create train and val ids
		train_idx = np.array([u * self.num_items + v for u, v, _ in graph_train])
		val_idx = np.array([u * self.num_items + v for u, v, _ in graph_val])

		u_train_idx, v_train_idx, train_labels = graph_train.T
		u_val_idx, v_val_idx, val_labels = graph_val.T

		# If testing environment
		if self.testing:
			u_train_idx = np.hstack([u_train_idx, u_val_idx])
			v_train_idx = np.hstack([v_train_idx, v_val_idx])
			train_labels = np.hstack([train_labels, val_labels])
			train_idx = np.hstack([train_idx, val_idx])

		# Make training adjacency matrix
		rating_mx_train = np.zeros(self.num_users * self.num_items, dtype=np.float32)
		rating_mx_train[train_idx] = self.labels[train_idx].astype(np.float32) + 1.
		rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(self.num_users, self.num_items))

		return rating_mx_train, \
			train_labels, u_train_idx, v_train_idx, \
			val_labels, u_val_idx, v_val_idx