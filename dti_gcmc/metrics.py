#!/usr/bin/python

import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

def softmax_accuracy(preds, labels):
    """
    Accuracy for multiclass model.
    :param preds: predictions
    :param labels: ground truth labels
    :return: average accuracy
    """
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.to_int64(labels))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)


def area_under_curve(preds, labels):
    """
    Area under curve for binary classification.
    :param preds: predictions
    :param labels: ground truth labels
    :return: area under curve
    """

    preds = tf.argmax(preds, 1)
    return tf.contrib.metrics.streaming_auc(labels, preds)


def expected_rmse(logits, labels, class_values=None):
    """
    Computes the root mean square error with the predictions
    computed as average predictions. Note that without the average
    this cannot be used as a loss function as it would not be differentiable.
    :param logits: predicted logits
    :param labels: ground truth label
    :param class_values: rating values corresponding to each class.
    :return: rmse
    """

    probs = tf.nn.softmax(logits)
    if class_values is None:
        scores = tf.to_float(tf.range(start=0, limit=logits.get_shape()[1]) + 1)
        y = tf.to_float(labels) + 1.  # assumes class values are 1, ..., num_classes
    else:
        scores = class_values
        y = tf.gather(class_values, labels)

    pred_y = tf.reduce_sum(probs * scores, 1)

    diff = tf.subtract(y, pred_y)
    exp_rmse = tf.square(diff)
    exp_rmse = tf.cast(exp_rmse, dtype=tf.float32)

    return tf.sqrt(tf.reduce_mean(exp_rmse))


def rmse(logits, labels, class_values=None):
    """
    Computes the mean square error with the predictions
    computed as average predictions. Note that without the average
    this cannot be used as a loss function as it would not be differentiable.
    :param logits: predicted logits
    :param labels: ground truth labels for the ratings, 1-D array containing 0-num_classes-1 ratings
    :param class_values: rating values corresponding to each class.
    :return: mse
    """

    if class_values is None:
        y = tf.to_float(labels) + 1.  # assumes class values are 1, ..., num_classes
    else:
        y = tf.gather(class_values, labels)

    pred_y = logits

    diff = tf.subtract(y, pred_y)
    mse = tf.square(diff)
    mse = tf.cast(mse, dtype=tf.float32)

    return tf.sqrt(tf.reduce_mean(mse))

def generate_weights(x, class_weights):
    """ Makes an replaces x according to its
        corresponding class weight. """

    if x == 0:
        return class_weights[0]

    return class_weights[1]

def softmax_cross_entropy(outputs, labels, class_weights, 
                                use_class_weights=False):
    """ computes average softmax cross entropy """

    weights = tf.map_fn(lambda x: generate_weights(x, class_weights), 
        labels, dtype=tf.float32)

    loss = tf.losses.sparse_softmax_cross_entropy(labels, outputs, 
        weights=weights)

    return tf.reduce_mean(loss)