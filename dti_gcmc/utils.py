from __future__ import division
from __future__ import print_function

import operator
import numpy as np
from collections import Counter

def user_based_probability_distribution(u_nodes,
										v_nodes,
										graph_0_train, 
										graph_1_train,
										threshold=10,
										boost=False):
	""" Generate a user based probability distribution. """
	
	users = Counter(graph_1_train.T[0])
	sorted_users = sorted(users.items(), 
						  key=operator.itemgetter(1))
	sorted_users.reverse()

	# Get users of interest
	print('Getting users of interest...')
	
	if threshold == 'all':
		print('Choosing all users...')
		users_of_interest = np.unique(u_nodes)
	else:
		print('Choosing top ' + str(threshold) + ' users...')
		users_of_interest = list()
		for user, freq in sorted_users:
			if freq >= threshold:
				users_of_interest.append(user)

	# Get a score for each movie
	print('Scoring movies...')
	movies_watched = dict()
	for user in users_of_interest:
		movies_watched[user] = []

	for u, v, r in graph_1_train:
		if u in users_of_interest:
			movies_watched[u].append(v)

	movies = dict()
	for i in np.unique(v_nodes):
		movies[i] = 0

	for k in movies_watched.keys():
		for movie in movies_watched[k]:
			movies[movie] += -1 * users[k]

	m = min(movies.values())
	for k in movies.keys():
		movies[k] += -1*m + 1
		
	if boost:
		print('Boosting values...')
		for k in movies.keys():
			movies[k] *= 10.
			
	# Generate probabilities
	print('Generating probabilities...')
	p = list()
	for u, v, r in graph_0_train:
		p.append(movies[v])

	norm_factor = sum(p)
	for i in range(len(p)):
		p[i] /= norm_factor

	adj_flag = False
	for i in range(len(p)):
		if p[0] - sum(p) + 1. > 0:
			p[0] += 1. - sum(p)
			adj_flag = True

	if not adj_flag:
		raise RuntimeError('Probabilities not adjusted to 1.')
		
	print('Done.')

	return np.array(p), movies


def construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero, v_features_nonzero,
						support, support_t, labels, u_indices, v_indices, class_values,
						dropout, u_features_side=None, v_features_side=None):
	"""
	Function that creates feed dictionary when running tensorflow sessions.
	"""

	feed_dict = dict()
	feed_dict.update({placeholders['u_features']: u_features})
	feed_dict.update({placeholders['v_features']: v_features})
	feed_dict.update({placeholders['u_features_nonzero']: u_features_nonzero})
	feed_dict.update({placeholders['v_features_nonzero']: v_features_nonzero})
	feed_dict.update({placeholders['support']: support})
	feed_dict.update({placeholders['support_t']: support_t})

	feed_dict.update({placeholders['labels']: labels})
	feed_dict.update({placeholders['user_indices']: u_indices})
	feed_dict.update({placeholders['item_indices']: v_indices})

	feed_dict.update({placeholders['dropout']: dropout})
	feed_dict.update({placeholders['class_values']: class_values})

	if (u_features_side is not None) and (v_features_side is not None):
		feed_dict.update({placeholders['u_features_side']: u_features_side})
		feed_dict.update({placeholders['v_features_side']: v_features_side})

	return feed_dict
