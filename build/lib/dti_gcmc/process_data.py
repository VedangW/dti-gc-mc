from __future__ import print_function

import pickle
import argparse
import numpy as np
import scipy.sparse as sp

from tqdm import tqdm
from scipy.io import loadmat


def get_data_from_txt_file(fpath, adj=False):
	""" Get data from the file fpath
		in needed format. """
	
	with open(fpath) as f:
		lines = f.readlines()
		
	data = list()
	for line in lines:
		line = line.strip().split('\t')
		line = np.array(line)
		data.append(line)

	if not adj:
		data = np.vstack(data).astype('float32')
	else:
		data = np.vstack(data).astype(int)
	
	return data

# Command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default='dataset_2',
				choices=['dataset_1', 'dataset_2'],
				help="Dataset string.")

ap.add_argument("-s", "--subset",
				help="Set true to subset.", action='store_true')

ap.add_argument("-v", "--verbose",
				help="Set true to print details of graph.", 
				action='store_true')

args = ap.parse_args()

# Global constants
DATASET = args.dataset
SUBSET = args.subset
VERBOSE = args.verbose

data_repo = 'data/dti_data/'
store_repo = 'data/dti_store/'

if DATASET == 'dataset_1':

	drug_feat = loadmat(data_repo + 'drugFeatureVectors.mat')
	target_feat = loadmat(data_repo + 'targetFeatureVectors.mat')
	interactions = loadmat(data_repo + 'interactionMatrix.mat')

	u_feat = drug_feat['drugFeatureVectors']
	v_feat = target_feat['targetFeatureVectors']
	weights = interactions['Y']

elif DATASET == 'dataset_2':

	u_feat_path = data_repo + 'drugFeatureVectors.txt'
	v_feat_path = data_repo + 'targetFeatureVectors.txt'
	weights_path = data_repo + 'interactionMatrix.txt'

	u_feat = get_data_from_txt_file(u_feat_path)
	v_feat = get_data_from_txt_file(v_feat_path)
	weights = get_data_from_txt_file(weights_path, True)


## Collecting records
u_nodes = list()
v_nodes = list()
ratings = list()

# Create records
for i in tqdm(range(len(weights)), 
	desc='Collecting records...', 
	unit='drugs'):
	for j in range(len(weights[i])):
		
		if weights[i][j] == 1:
			u_nodes.append(i)
			v_nodes.append(j)
			ratings.append(1)
			
		else:
			u_nodes.append(i)
			v_nodes.append(j)
			ratings.append(0)
			
# Convert to numpy arrays
u_nodes = np.array(u_nodes)
v_nodes = np.array(v_nodes)
ratings = np.array(ratings)

num_u = u_feat.shape[0] 
num_v = v_feat.shape[0]

## Preprocessing

# Create graph
graph = [u_nodes, v_nodes, ratings]
graph = np.vstack(graph)

if SUBSET:
	graph = graph.T

	# Sort based on ratings and split for 0 and 1 ratings
	graph = graph[graph[:, 2].argsort()]
	graph_0, graph_1 = np.split(graph, np.where(np.diff(graph[:,2]))[0]+1)

	# Randomly select only 100000 from 0s
	index = np.random.choice(graph_0.shape[0], 100000, replace=False)
	graph_0 = graph_0[index]

	# Recombine graph
	graph = np.concatenate([graph_0, graph_1], axis=0)
	np.random.shuffle(graph)

	graph = graph.T

else:
	# Shuffle data in graph
	graph = graph.T
	np.random.shuffle(graph)
	graph = graph.T

# Get data format back from graph
u_nodes, v_nodes, ratings = graph[0], graph[1], graph[2]

# Convert features to sparse matrices
u_feat = sp.csr_matrix(u_feat)
v_feat = sp.csr_matrix(v_feat)

# Combine as a tuple
graph_data = (num_u, num_v, u_nodes, v_nodes, ratings, u_feat, v_feat)

if VERBOSE:
	print ('No. of drugs =', num_u)
	print ('No. of targets =', num_v)
	print ('Total no. of interactions =', u_nodes.shape[0])

print ('')
print ('Saving graph...')

if DATASET == 'dataset_1':
	with open(store_repo + 'graph_1.pkl', 'w') as f:
		pickle.dump(graph_data, f)
elif DATASET == 'dataset_2':
	with open(store_repo + 'graph_2.pkl', 'w') as f:
		pickle.dump(graph_data, f)

print ('Done.')