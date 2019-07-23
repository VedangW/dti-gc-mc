from __future__ import division
from __future__ import print_function

import sys
import time
import json
import pickle
import argparse
import datetime

import numpy as np
import tensorflow as tf
import scipy.sparse as sp

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score

from preprocessing import create_trainvaltest_split, \
    sparse_to_tuple, preprocess_user_item_features, globally_normalize_bipartite_adjacency, \
    normalize_features
from model import RecommenderGAE, RecommenderSideInfoGAE
from utils import construct_feed_dict

# Set random seed
# seed = 123 # use only for unit testing
seed = int(time.time())
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default='data_2_small',
                choices=['data_1', 'data_2', 'data_1_small', 'data_2_small'],
                help="Dataset string.")

ap.add_argument("-lr", "--learning_rate", type=float, default=0.01,
                help="Learning rate")

ap.add_argument("-e", "--epochs", type=int, default=20,
                help="Number training epochs")

ap.add_argument("-hi", "--hidden", type=int, nargs=4, default=[512, 128, 64, 32],
                help="Number hidden units in 1st and 2nd layer")

ap.add_argument("-fhi", "--feat_hidden", type=int, default=64,
                help="Number hidden units in the dense layer for features")

ap.add_argument("-ac", "--accumulation", type=str, default="stack", choices=['sum', 'stack'],
                help="Accumulation function: sum or stack.")

ap.add_argument("-do", "--dropout", type=float, default=0.7,
                help="Dropout fraction")

ap.add_argument("-nb", "--num_basis_functions", type=int, default=2,
                help="Number of basis functions for Mixture Model GCN.")

ap.add_argument("-ds", "--data_seed", type=int, default=1234,
                help="""Seed used to shuffle data in data_utils, taken from cf-nade (1234, 2341, 3412, 4123, 1324).
                     Only used for ml_1m and ml_10m datasets.""")

ap.add_argument("-sdir", "--summaries_dir", type=str, default='logs/' + str(datetime.datetime.now()).replace(' ', '_'),
                help="Directory for saving tensorflow summaries.")

ap.add_argument("-act", "--activation", type=str, default='relu', 
                choices=['relu', 'relu6', 'tanh', 'sigmoid'],
                help="Activation function used for StackGCN, OrdinalGCN and Dense layers.")

ap.add_argument("-r", "--regularization", type=str, default='none',
                choices=['none', 'denoising', 'sparse', 'derivative'],
                help="Regularization for the model.")

ap.add_argument("-st", "--split_type", type=str, default='stratified',
                choices=['random', 'stratified'],
                help="Type of split for train-test split.")

ap.add_argument("-cw", "--class_weights", type=float, nargs=2, default=[1., 25.],
                help="The class weights for binary classification")

ap.add_argument("-sht", "--show_test_results",
                help="Callback to show continuous results on the test set.", 
                action='store_true')

ap.add_argument("-acw", "--automatic_class_weights",
                help="Generate class weights automatically.", 
                action='store_true')

# Boolean flags
fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-nsym', '--norm_symmetric', dest='norm_symmetric',
                help="Option to turn on symmetric global normalization", action='store_true')
fp.add_argument('-nleft', '--norm_left', dest='norm_symmetric',
                help="Option to turn on left global normalization", action='store_false')
ap.set_defaults(norm_symmetric=True)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-f', '--features', dest='features',
                help="Whether to use features (1) or not (0)", action='store_true')
fp.add_argument('-no_f', '--no_features', dest='features',
                help="Whether to use features (1) or not (0)", action='store_false')
ap.set_defaults(features=False)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-ws', '--write_summary', dest='write_summary',
                help="Option to turn on summary writing", action='store_true')
fp.add_argument('-no_ws', '--no_write_summary', dest='write_summary',
                help="Option to turn off summary writing", action='store_false')
ap.set_defaults(write_summary=False)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-t', '--testing', dest='testing',
                help="Option to turn on test set evaluation", action='store_true')
fp.add_argument('-v', '--validation', dest='testing',
                help="Option to only use validation set evaluation", action='store_false')
ap.set_defaults(testing=False)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-ucw', '--use_class_weights', dest='use_class_weights',
                help="Set to use class weights strategy", action='store_true')
fp.add_argument('-ovs', '--oversample', dest='use_class_weights',
                help="Set to use oversampling", action='store_false')
ap.set_defaults(use_class_weights=False)


args = vars(ap.parse_args())

print('Settings:')
print(args, '\n')

# Activation function choices
activations = {'relu': tf.nn.relu,
                'relu6': tf.nn.relu6,
                'sigmoid': tf.sigmoid,
                'tanh': tf.tanh}

# Define parameters
DATASET = args['dataset']
DATASEED = args['data_seed']
NB_EPOCH = args['epochs']
DO = args['dropout']
HIDDEN = args['hidden']
FEATHIDDEN = args['feat_hidden']
BASES = args['num_basis_functions']
LR = args['learning_rate']
WRITESUMMARY = args['write_summary']
SUMMARIESDIR = args['summaries_dir']
FEATURES = args['features']
SYM = args['norm_symmetric']
TESTING = args['testing']
ACCUM = args['accumulation']
ACTIVATION = activations[args['activation']]
REG = args['regularization']
STYPE = args['split_type']
USE_CLASS_WEIGHTS = args['use_class_weights']
CLASS_WEIGHTS = args['class_weights']
SHOWTEST = args['show_test_results']
AUTO_CW = args['automatic_class_weights']

CLASS_WEIGHTS = np.array(CLASS_WEIGHTS).astype('float32')

SELFCONNECTIONS = False
SPLITFROMFILE = False
VERBOSE = True

NUMCLASSES = 2

# Splitting dataset in training, validation and test set

if DATASET == 'data_1':
    datasplit_path = 'data/' + DATASET + '/data_1.pickle'
elif DATASET == 'data_2':
    datasplit_path = 'data/' + DATASET + '/data_2.pickle'
elif DATASET == 'data_1_small':
    datasplit_path = 'data/' + DATASET + '/data_1_small.pickle'
elif DATASET == 'data_2_small':
    datasplit_path = 'data/' + DATASET + '/data_2_small.pickle'
else:
    raise ValueError('Dataset not recognized.')


u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
    val_labels, val_u_indices, val_v_indices, test_labels, \
    test_u_indices, test_v_indices, class_values, class_weights = create_trainvaltest_split(DATASET, DATASEED, TESTING,
                                                                             datasplit_path, SPLITFROMFILE,
                                                                             STYPE, VERBOSE)

if AUTO_CW:
    CLASS_WEIGHTS = class_weights
    print ('Using auto-generated class weights.')

num_users, num_items = adj_train.shape

num_side_features = 0

# feature loading
if not FEATURES:
    u_features = sp.identity(num_users, format='csr')
    v_features = sp.identity(num_items, format='csr')

    u_features, v_features = preprocess_user_item_features(u_features, v_features)

elif FEATURES and u_features is not None and v_features is not None:
    # use features as side information and node_id's as node input features

    print("Normalizing feature vectors...")
    u_features_side = normalize_features(u_features)
    v_features_side = normalize_features(v_features)

    u_features_side, v_features_side = preprocess_user_item_features(u_features_side, v_features_side)

    u_features_side = np.array(u_features_side.todense(), dtype=np.float32)
    v_features_side = np.array(v_features_side.todense(), dtype=np.float32)

    num_side_features = u_features_side.shape[1]

    # node id's for node input features
    id_csr_v = sp.identity(num_items, format='csr')
    id_csr_u = sp.identity(num_users, format='csr')

    u_features, v_features = preprocess_user_item_features(id_csr_u, id_csr_v)

else:
    raise ValueError('Features flag is set to true but no features are loaded from dataset ' + DATASET)


# global normalization
support = []
support_t = []
adj_train_int = sp.csr_matrix(adj_train, dtype=np.int32)

for i in range(NUMCLASSES):
    # build individual binary rating matrices (supports) for each rating
    support_unnormalized = sp.csr_matrix(adj_train_int == i + 1, dtype=np.float32)

    if support_unnormalized.nnz == 0 and DATASET != 'yahoo_music':
        # yahoo music has dataset split with not all ratings types present in training set.
        # this produces empty adjacency matrices for these ratings.
        sys.exit('ERROR: normalized bipartite adjacency matrix has only zero entries!!!!!')

    support_unnormalized_transpose = support_unnormalized.T
    support.append(support_unnormalized)
    support_t.append(support_unnormalized_transpose)


support = globally_normalize_bipartite_adjacency(support, symmetric=SYM)
support_t = globally_normalize_bipartite_adjacency(support_t, symmetric=SYM)

if SELFCONNECTIONS:
    support.append(sp.identity(u_features.shape[0], format='csr'))
    support_t.append(sp.identity(v_features.shape[0], format='csr'))

num_support = len(support)
support = sp.hstack(support, format='csr')
support_t = sp.hstack(support_t, format='csr')

if ACCUM == 'stack':
    div = HIDDEN[0] // num_support
    if HIDDEN[0] % num_support != 0:
        print("""\nWARNING: HIDDEN[0] (=%d) of stack layer is adjusted to %d such that
                  it can be evenly split in %d splits.\n""" % (HIDDEN[0], num_support * div, num_support))
    HIDDEN[0] = num_support * div

# Collect all user and item nodes for test set
test_u = list(set(test_u_indices))
test_v = list(set(test_v_indices))
test_u_dict = {n: i for i, n in enumerate(test_u)}
test_v_dict = {n: i for i, n in enumerate(test_v)}

test_u_indices = np.array([test_u_dict[o] for o in test_u_indices])
test_v_indices = np.array([test_v_dict[o] for o in test_v_indices])

test_support = support[np.array(test_u)]
test_support_t = support_t[np.array(test_v)]

# Collect all user and item nodes for validation set
val_u = list(set(val_u_indices))
val_v = list(set(val_v_indices))
val_u_dict = {n: i for i, n in enumerate(val_u)}
val_v_dict = {n: i for i, n in enumerate(val_v)}

val_u_indices = np.array([val_u_dict[o] for o in val_u_indices])
val_v_indices = np.array([val_v_dict[o] for o in val_v_indices])

val_support = support[np.array(val_u)]
val_support_t = support_t[np.array(val_v)]

# Collect all user and item nodes for train set
train_u = list(set(train_u_indices))
train_v = list(set(train_v_indices))
train_u_dict = {n: i for i, n in enumerate(train_u)}
train_v_dict = {n: i for i, n in enumerate(train_v)}

train_u_indices = np.array([train_u_dict[o] for o in train_u_indices])
train_v_indices = np.array([train_v_dict[o] for o in train_v_indices])

train_support = support[np.array(train_u)]
train_support_t = support_t[np.array(train_v)]

# features as side info
if FEATURES:
    test_u_features_side = u_features_side[np.array(test_u)]
    test_v_features_side = v_features_side[np.array(test_v)]

    val_u_features_side = u_features_side[np.array(val_u)]
    val_v_features_side = v_features_side[np.array(val_v)]

    train_u_features_side = u_features_side[np.array(train_u)]
    train_v_features_side = v_features_side[np.array(train_v)]

else:
    test_u_features_side = None
    test_v_features_side = None

    val_u_features_side = None
    val_v_features_side = None

    train_u_features_side = None
    train_v_features_side = None

placeholders = {
    'u_features': tf.sparse_placeholder(tf.float32, shape=np.array(u_features.shape, dtype=np.int64)),
    'v_features': tf.sparse_placeholder(tf.float32, shape=np.array(v_features.shape, dtype=np.int64)),
    'u_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'v_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'labels': tf.placeholder(tf.int32, shape=(None,)),

    'u_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),
    'v_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),

    'user_indices': tf.placeholder(tf.int32, shape=(None,)),
    'item_indices': tf.placeholder(tf.int32, shape=(None,)),

    'class_values': tf.placeholder(tf.float32, shape=class_values.shape),

    'dropout': tf.placeholder_with_default(0., shape=()),
    'weight_decay': tf.placeholder_with_default(0., shape=()),

    'support': tf.sparse_placeholder(tf.float32, shape=(None, None)),
    'support_t': tf.sparse_placeholder(tf.float32, shape=(None, None)),
}

# create model
if FEATURES:
    print ("Using RecommenderSideInfoGAE...")
    model = RecommenderSideInfoGAE(placeholders,
                                   input_dim=u_features.shape[1],
                                   feat_hidden_dim=FEATHIDDEN,
                                   num_classes=NUMCLASSES,
                                   num_support=num_support,
                                   self_connections=SELFCONNECTIONS,
                                   num_basis_functions=BASES,
                                   hidden=HIDDEN,
                                   num_users=num_users,
                                   num_items=num_items,
                                   accum=ACCUM,
                                   activation_function=ACTIVATION,
                                   learning_rate=LR,
                                   num_side_features=num_side_features,
                                   regularization=REG,
                                   use_class_weights=USE_CLASS_WEIGHTS,
                                   class_weights=CLASS_WEIGHTS,
                                   logging=True)
else:
    print ("Using RecommenderGAE...")
    model = RecommenderGAE(placeholders,
                           input_dim=u_features.shape[1],
                           num_classes=NUMCLASSES,
                           num_support=num_support,
                           self_connections=SELFCONNECTIONS,
                           num_basis_functions=BASES,
                           hidden=HIDDEN,
                           num_users=num_users,
                           num_items=num_items,
                           accum=ACCUM,
                           learning_rate=LR,
                           activation_function=ACTIVATION,
                           regularization=REG,
                           use_class_weights=USE_CLASS_WEIGHTS,
                           class_weights=CLASS_WEIGHTS,
                           logging=True)

# Convert sparse placeholders to tuples to construct feed_dict
test_support = sparse_to_tuple(test_support)
test_support_t = sparse_to_tuple(test_support_t)

val_support = sparse_to_tuple(val_support)
val_support_t = sparse_to_tuple(val_support_t)

train_support = sparse_to_tuple(train_support)
train_support_t = sparse_to_tuple(train_support_t)

u_features = sparse_to_tuple(u_features)
v_features = sparse_to_tuple(v_features)
assert u_features[2][1] == v_features[2][1], 'Number of features of users and items must be the same!'

num_features = u_features[2][1]
u_features_nonzero = u_features[1].shape[0]
v_features_nonzero = v_features[1].shape[0]

# Feed_dicts for validation and test set stay constant over different update steps
train_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                      v_features_nonzero, train_support, train_support_t,
                                      train_labels, train_u_indices, train_v_indices, class_values, DO,
                                      train_u_features_side, train_v_features_side)
# No dropout for validation and test runs
val_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                    v_features_nonzero, val_support, val_support_t,
                                    val_labels, val_u_indices, val_v_indices, class_values, 0.,
                                    val_u_features_side, val_v_features_side)

test_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                     v_features_nonzero, test_support, test_support_t,
                                     test_labels, test_u_indices, test_v_indices, class_values, 0.,
                                     test_u_features_side, test_v_features_side)


print ('num_support =', num_support)

# Collect all variables to be logged into summary
merged_summary = tf.summary.merge_all()

sess = tf.Session()

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)

if WRITESUMMARY:
    train_summary_writer = tf.summary.FileWriter(SUMMARIESDIR + '/train', sess.graph)
    val_summary_writer = tf.summary.FileWriter(SUMMARIESDIR + '/val')
else:
    train_summary_writer = None
    val_summary_writer = None

best_val_score = np.inf
best_val_loss = np.inf
best_val_acc = 0.
best_val_auc = 0.
best_epoch = 0
best_acc_epoch = 0
wait = 0
total_time = 0.

print('Training...')

for epoch in range(NB_EPOCH):

    t = time.time()

    # Run single weight update
    # outs = sess.run([model.opt_op, model.loss, model.rmse], feed_dict=train_feed_dict)
    # with exponential moving averages
    outs = sess.run([model.training_op, model.loss, model.rmse], feed_dict=train_feed_dict)

    train_avg_loss = outs[1]
    train_rmse = outs[2]

    val_avg_loss, val_rmse, val_accuracy, auc, outputs, labels = sess.run([model.loss, 
                                                                              model.rmse, 
                                                                              model.accuracy, 
                                                                              model.auc,
                                                                              model.outputs, 
                                                                              model.labels], 
                                                                              feed_dict=val_feed_dict)

    val_auc, update_op = auc
    outputs = outputs[:, 1]

    epoch_time = time.time() - t
    total_time += epoch_time

    sk_val_auc = roc_auc_score(labels, outputs)

    if SHOWTEST:
        test_avg_loss, test_rmse, test_accuracy, test_auc_with_op, test_outputs, test_labels = sess.run([model.loss, 
                                                                              model.rmse, 
                                                                              model.accuracy, 
                                                                              model.auc, 
                                                                              model.outputs, 
                                                                              model.labels], 
                                                                              feed_dict=test_feed_dict)

        test_auc, update_op = test_auc_with_op
        test_outputs = test_outputs[:, 1]
        sk_test_auc = roc_auc_score(test_labels, test_outputs)

    if SHOWTEST and VERBOSE:
        print("[*] Epoch:", '%04d' % epoch,
              "val_auc=", "{:.5f}".format(val_auc),
              "test_auc=", "{:.5f}".format(test_auc),
              "sk_val_auc=", "{:.5f}".format(sk_val_auc),
              "sk_test_auc=", "{:.5f}".format(sk_test_auc),
              # "tp=", tp, "fp=", fp, "tn=", tn, "fn=", fn, 
              "\ttime=", "{:.5f}".format(time.time() - t))

    elif (not SHOWTEST) and VERBOSE:
        print("[*] Epoch:", '%04d' % epoch,
              "train_loss=", "{:.5f}".format(train_avg_loss),
              "val_loss=", "{:.5f}".format(val_avg_loss),
              "val_auc=", "{:.5f}".format(val_auc),
              "sk_val_auc=", "{:.5f}".format(sk_val_auc),
              "\t\ttime=", "{:.5f}".format(time.time() - t))

    if val_rmse < best_val_score:
        best_val_score = val_rmse
        best_epoch = epoch

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        best_acc_epoch = epoch

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_auc_epoch = epoch

    if epoch % 20 == 0 and WRITESUMMARY:
        # Train set summary
        summary = sess.run(merged_summary, feed_dict=train_feed_dict)
        train_summary_writer.add_summary(summary, epoch)
        train_summary_writer.flush()

        # Validation set summary
        summary = sess.run(merged_summary, feed_dict=val_feed_dict)
        val_summary_writer.add_summary(summary, epoch)
        val_summary_writer.flush()

    if epoch % 100 == 0 and epoch > 1000 and not TESTING and False:
        saver = tf.train.Saver()
        save_path = saver.save(sess, "tmp/%s_seed%d.ckpt" % (model.name, DATASEED), global_step=model.global_step)

        # load polyak averages
        variables_to_restore = model.variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, save_path)

        val_avg_loss, val_rmse = sess.run([model.loss, model.rmse], feed_dict=val_feed_dict)

        print('polyak val loss = ', val_avg_loss)
        print('polyak val rmse = ', val_rmse)

        # Load back normal variables
        saver = tf.train.Saver()
        saver.restore(sess, save_path)


# store model including exponential moving averages
saver = tf.train.Saver()
save_path = saver.save(sess, "tmp/%s.ckpt" % model.name, global_step=model.global_step)


if VERBOSE:
    print("\nOptimization Finished!")
    print('best validation score =', best_val_score, 'at iteration', best_epoch)
    print('best validation accuracy =', best_val_acc, 'at iteration', best_acc_epoch)
    print('best validation auc =', best_val_auc, 'at iteration', best_auc_epoch)
    print('total time taken = ' + str(total_time) + 's')


if TESTING:
    test_avg_loss, test_rmse, test_accuracy, test_auc_with_op, outputs, labels = sess.run([model.loss, 
                                                                                  model.rmse, 
                                                                                  model.accuracy, 
                                                                                  model.auc, 
                                                                                  model.outputs, 
                                                                                  model.labels], 
                                                                                  feed_dict=test_feed_dict)

    test_auc, update_op = test_auc_with_op

    outputs = outputs[:, 1]
    sk_test_auc = roc_auc_score(labels, outputs)

    print('')
    print('test loss = ', test_avg_loss)
    print('test rmse = ', test_rmse)
    print('test accuracy =', test_accuracy)
    print('test auc =', sk_test_auc)
    print('test update_op =', update_op)

    print('Saving data...')
    with open('data/' + DATASET + '/pred_and_labels.pkl', 'w') as f:
        pickle.dump([outputs, labels], f)

    # restore with polyak averages of parameters
    variables_to_restore = model.variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, save_path)

    test_avg_loss, test_rmse = sess.run([model.loss, model.rmse], feed_dict=test_feed_dict)
    print('polyak test loss = ', test_avg_loss)
    print('polyak test rmse = ', test_rmse)

else:
    # restore with polyak averages of parameters
    variables_to_restore = model.variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, save_path)

    val_avg_loss, val_rmse = sess.run([model.loss, model.rmse], feed_dict=val_feed_dict)
    print('polyak val loss = ', val_avg_loss)
    print('polyak val rmse = ', val_rmse)

print('\nSETTINGS:\n')
for key, val in sorted(vars(ap.parse_args()).iteritems()):
    print(key, val)

print('global seed = ', seed)

# For parsing results from file
results = vars(ap.parse_args()).copy()
results.update({'best_val_score': float(best_val_score), 'best_epoch': best_epoch})
print(json.dumps(results))

sess.close()