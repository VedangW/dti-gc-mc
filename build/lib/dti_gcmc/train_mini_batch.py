from __future__ import division
from __future__ import print_function

import time
import json
import argparse
import datetime

import numpy as np
import tensorflow as tf
import scipy.sparse as sp

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score

from preprocessing import create_trainvaltest_split, \
    sparse_to_tuple, preprocess_user_item_features, globally_normalize_bipartite_adjacency
from model import RecommenderGAE
from utils import construct_feed_dict
from data_utils import data_iterator


# Set random seed
# seed = 123 # use only for unit testing
seed = int(time.time())
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="dataset_2", 
                choices=['dataset_1', 'dataset_2'],
                help="Dataset string.")

ap.add_argument("-lr", "--learning_rate", type=float, default=0.01,
                help="Learning rate")

ap.add_argument("-e", "--epochs", type=int, default=20,
                help="Number training epochs")

ap.add_argument("-hi", "--hidden", type=int, nargs=2, default=[512, 64],
                help="Number hidden units in 1st and 2nd layer")

ap.add_argument("-ac", "--accumulation", type=str, default="stack", choices=['sum', 'stack'],
                help="Accumulation function: sum or stack.")

ap.add_argument("-do", "--dropout", type=float, default=0.3,
                help="Dropout fraction")

ap.add_argument("-edo", "--edge_dropout", type=float, default=0.,
                help="Edge dropout rate (1 - keep probability).")

ap.add_argument("-nb", "--num_basis_functions", type=int, default=2,
                help="Number of basis functions for Mixture Model GCN.")

ap.add_argument("-ds", "--data_seed", type=int, default=1234,
                help="Seed used to shuffle data in data_utils, taken from cf-nade (1234, 2341, 3412, 4123, 1324)")

ap.add_argument("-sdir", "--summaries_dir", type=str, default='logs/' + str(datetime.datetime.now()).replace(' ', '_'),
                help="Dataset string ('ml_100k', 'ml_1m')")

ap.add_argument("-bs", "--batch_size", type=int, default=10000,
                help="Batch size used for batching loss function contributions.")

ap.add_argument("-act", "--activation", type=str, default='relu', 
                choices=['relu', 'relu6', 'tanh', 'sigmoid'],
                help="Activation function used for StackGCN, OrdinalGCN and Dense layers.")

ap.add_argument("-r", "--regularization", type=str, default='none',
                choices=['none', 'denoising', 'sparse', 'derivative'],
                help="Regularization for the model.")

ap.add_argument("-st", "--split_type", type=str, default='stratified',
                choices=['random', 'stratified', 'stratified_with_weights'],
                help="Type of split for train-test split.")

ap.add_argument("-cw", "--class_weights", type=float, nargs=2, default=[1., 25.],
                help="The class weights for binary classification.")

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
BASES = args['num_basis_functions']
LR = args['learning_rate']
WRITESUMMARY = args['write_summary']
SUMMARIESDIR = args['summaries_dir']
FEATURES = args['features']
TESTING = args['testing']
BATCHSIZE = args['batch_size']
SYM = args['norm_symmetric']
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

if DATASET == 'dataset_1':
    datasplit_path = 'data/' + DATASET + '/dataset_1.pickle'
elif DATASET == 'dataset_2':
    datasplit_path = 'data/' + DATASET + '/dataset_2.pickle'
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

# num_mini_batch = np.int(np.ceil(train_labels.shape[0]/float(BATCHSIZE)))
num_mini_batch = train_labels.shape[0]//BATCHSIZE
print ('num mini batch = ', num_mini_batch)

num_users, num_items = adj_train.shape

# feature loading
if not FEATURES:
    u_features = sp.identity(num_users, format='csr')
    v_features = sp.identity(num_items, format='csr')

    u_features, v_features = preprocess_user_item_features(u_features, v_features)

else:
    raise ValueError('Features are not supported in this implementation.')

# global normalization
support = []
support_t = []
adj_train_int = sp.csr_matrix(adj_train, dtype=np.int32)
for i in range(NUMCLASSES):
    # build individual binary rating matrices (supports) for each rating
    support_unnormalized = sp.csr_matrix(adj_train_int == i + 1, dtype=np.float32)
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

placeholders = {
    'u_features': tf.sparse_placeholder(tf.float32, shape=np.array(u_features.shape, dtype=np.int64)),
    'v_features': tf.sparse_placeholder(tf.float32, shape=np.array(v_features.shape, dtype=np.int64)),
    'u_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'v_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'labels': tf.placeholder(tf.int32, shape=(None,)),

    'user_indices': tf.placeholder(tf.int32, shape=(None,)),
    'item_indices': tf.placeholder(tf.int32, shape=(None,)),

    'dropout': tf.placeholder_with_default(0., shape=()),

    'class_values': tf.placeholder(tf.float32, shape=class_values.shape),

    'support': tf.sparse_placeholder(tf.float32, shape=(None, None)),
    'support_t': tf.sparse_placeholder(tf.float32, shape=(None, None)),
}

# create model
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


u_features = sparse_to_tuple(u_features)
v_features = sparse_to_tuple(v_features)
assert u_features[2][1] == v_features[2][1], 'Number of features of users and items must be the same!'

num_features = u_features[2][1]
u_features_nonzero = u_features[1].shape[0]
v_features_nonzero = v_features[1].shape[0]

# Feed_dicts for validation and test set stay constant over different update steps
# No dropout for validation and test runs
val_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                    v_features_nonzero, val_support, val_support_t,
                                    val_labels, val_u_indices, val_v_indices, class_values, 0.)

test_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                     v_features_nonzero, test_support, test_support_t,
                                     test_labels, test_u_indices, test_v_indices, class_values, 0.)

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

    batch_iter = 0
    data_iter = data_iterator([train_u_indices, train_v_indices, train_labels], batch_size=BATCHSIZE)

    try:
        while True:
            t = time.time()

            train_u_indices_batch, train_v_indices_batch, train_labels_batch = data_iter.next()

            # Collect all user and item nodes for train set
            train_u = list(set(train_u_indices_batch))
            train_v = list(set(train_v_indices_batch))
            train_u_dict = {n: i for i, n in enumerate(train_u)}
            train_v_dict = {n: i for i, n in enumerate(train_v)}

            train_u_indices_batch = np.array([train_u_dict[o] for o in train_u_indices_batch])
            train_v_indices_batch = np.array([train_v_dict[o] for o in train_v_indices_batch])

            train_support_batch = sparse_to_tuple(support[np.array(train_u)])
            train_support_t_batch = sparse_to_tuple(support_t[np.array(train_v)])

            train_feed_dict_batch = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                                        v_features_nonzero,
                                                        train_support_batch,
                                                        train_support_t_batch,
                                                        train_labels_batch, train_u_indices_batch,
                                                        train_v_indices_batch, class_values, DO)

            # with exponential moving averages
            outs = sess.run([model.training_op, model.loss, model.rmse], feed_dict=train_feed_dict_batch)

            train_avg_loss = outs[1]
            train_rmse = outs[2]

            val_avg_loss, val_rmse, val_accuracy, auc, outputs, labels = sess.run([model.loss, 
                                                                              model.rmse, 
                                                                              model.accuracy, 
                                                                              model.auc,
                                                                              model.outputs, 
                                                                              model.labels], 
                                                                              feed_dict=val_feed_dict)

            if SHOWTEST:
                test_avg_loss, test_rmse, test_accuracy, test_auc_with_op, test_outputs, test_labels = sess.run([model.loss, 
                                                                                      model.rmse, 
                                                                                      model.accuracy, 
                                                                                      model.auc, 
                                                                                      model.outputs, 
                                                                                      model.labels], 
                                                                                      feed_dict=test_feed_dict)

                test_auc, update_op = test_auc_with_op

            val_auc, update_op = auc
            
            iteration_time = time.time() - t
            total_time += iteration_time

            if SHOWTEST:
                test_outputs = sess.run(tf.argmax(test_outputs, 1))

                # tp, fp, tn, fn = 0, 0, 0, 0
                # for i in range(len(test_outputs)):
                #     if test_outputs[i] == 1 and test_labels[i] == 1:
                #         tp += 1
                #     elif test_outputs[i] == 1 and test_labels[i] != 1:
                #         fp += 1
                #     elif test_outputs[i] == 0 and test_labels[i] == 0:
                #         tn += 1
                #     elif test_outputs[i] == 0 and test_labels[i] != 0:
                #         fn += 1

                sk_test_auc = roc_auc_score(test_labels, test_outputs)

            if SHOWTEST and VERBOSE:
                print('[*] Iter: %04d' % (epoch*num_mini_batch + batch_iter),  " Epoch:", '%04d' % epoch,
                      "minibatch iter:", '%04d' % batch_iter,
                      "val_auc=", "{:.5f}".format(val_auc),
                      "test_auc=", "{:.5f}".format(test_auc),
                      "sk_test_auc=", "{:.5f}".format(sk_test_auc),
                      # "tp=", tp, "fp=", fp, "tn=", tn, "fn=", fn, 
                      "\ttime=", "{:.5f}".format(time.time() - t))

            elif (not SHOWTEST) and VERBOSE:
                print('[*] Iter: %04d' % (epoch*num_mini_batch + batch_iter),  " Epoch:", '%04d' % epoch,
                      "minibatch iter:", '%04d' % batch_iter,
                      "train_loss=", "{:.5f}".format(train_avg_loss),
                      "val_loss=", "{:.5f}".format(val_avg_loss),
                      "val_auc=", "{:.5f}".format(val_auc),
                      "\t\ttime=", "{:.5f}".format(time.time() - t))

            if val_rmse < best_val_score:
                best_val_score = val_rmse
                best_epoch = epoch*num_mini_batch + batch_iter

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_acc_epoch = epoch*num_mini_batch + batch_iter

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_auc_epoch = epoch*num_mini_batch + batch_iter

            if batch_iter % 20 == 0 and WRITESUMMARY:
                # Train set summary
                summary = sess.run(merged_summary, feed_dict=train_feed_dict_batch)
                train_summary_writer.add_summary(summary, epoch*num_mini_batch+batch_iter)
                train_summary_writer.flush()

                # Validation set summary
                summary = sess.run(merged_summary, feed_dict=val_feed_dict)
                val_summary_writer.add_summary(summary, epoch*num_mini_batch+batch_iter)
                val_summary_writer.flush()

            if epoch*num_mini_batch+batch_iter % 100 == 0 and not TESTING and False:
                saver = tf.train.Saver()
                save_path = saver.save(sess, "tmp/%s_seed%d.ckpt" % (model.name, DATASEED), global_step=model.global_step)

                # load polyak averages
                variables_to_restore = model.variable_averages.variables_to_restore()
                saver = tf.train.Saver(variables_to_restore)
                saver.restore(sess, save_path)

                val_avg_loss, val_rmse = sess.run([model.loss, model.rmse], feed_dict=val_feed_dict)

                print('polyak val loss = ', val_avg_loss)
                print('polyak val rmse = ', val_rmse)

                # load back normal variables
                saver = tf.train.Saver()
                saver.restore(sess, save_path)

            batch_iter += 1

    except StopIteration:
        pass

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

    print('')
    print('test loss = ', test_avg_loss)
    print('test rmse = ', test_rmse)
    print('test accuracy =', test_accuracy)
    print('test auc =', test_auc)
    print('test update_op =', update_op)

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
