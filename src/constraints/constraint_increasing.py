# Monotonically increasing
# f(x1) <= f(x2), for x2 = x1 + eps*v,
# where v[i] == 0 for i != k and v[i] == 1 for i == k
# and k is the variable for which the monotonicity is checked
import numpy as np
import copy
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from constraints.constraint import generate_xi
from SRConstraints import Constraint


def generate_samples(data, constr:Constraint):
    """
    Generates samples from data_range.
    nSamples is the total number of samples that should be equally distributed among all intervals.
    """
    labelBase = constr.name
    eps = constr.args['eps']
    varId = constr.args['var']
    n = int(constr.nbOfSamples/(len(constr.domain))) # --- samples per interval
    # ---
    data_x = [np.array([generate_xi(lb=d[0], ub=d[1]-eps, nSamples=n) for d in i]).T for i in constr.domain]
    data_x = np.concatenate(data_x, axis=0)
    # --- Data x[varId]+eps
    data_delta = copy.deepcopy(data_x)
    data_delta[:, varId] = data_delta[:, varId] + eps
    # ---
    data.forwardpass_data_boundaries[labelBase+'_x'] = (data.forwardpass_data.shape[0], data.forwardpass_data.shape[0] + data_x.shape[0] - 1)
    data.forwardpass_counts[labelBase+'_x'] = data_x.shape[0]
    data.forwardpass_data = np.append(data.forwardpass_data, data_x, axis=0)
    data.forwardpass_data_boundaries[labelBase+'_delta'] = (data.forwardpass_data.shape[0], data.forwardpass_data.shape[0] + data_delta.shape[0] - 1)
    data.forwardpass_counts[labelBase+'_delta'] = data_delta.shape[0]
    data.forwardpass_data = np.append(data.forwardpass_data, data_delta, axis=0)
    return data


def update_samples(data, constr:Constraint):
    """
    TODO
    Updates constraint samples.
    nSamples is the total number of samples that should be equally distributed among all intervals.
    """
    pass


def get_constraint_term(data, constr:Constraint, y_hat, weight=1.0):
    if weight == 0.0:
        return tf.constant(0.0, dtype=np.float64)
    # ---
    labelBase = constr.name
    # ---
    y_hatA = tf.slice(y_hat, [data.forwardpass_data_boundaries[labelBase+'_x'][0], 0], [data.forwardpass_counts[labelBase+'_x'], 1])
    y_hatB = tf.slice(y_hat, [data.forwardpass_data_boundaries[labelBase+'_delta'][0], 0], [data.forwardpass_counts[labelBase+'_delta'], 1])
    eps = constr.args['eps']
    minIncrease = eps * np.tan(0.0000174533)      # --- min angle: 0.00174533 rad == 0.1 deg
    penalty = np.float64(0.01)
    # --- Calculate loss
    increase = tf.math.maximum(tf.subtract(y_hatB, y_hatA), 0.)
    tf_penalty = tf.where(increase < minIncrease, minIncrease, 0.0)     # --- penalty
    tf_error = tf.add(tf.math.maximum(tf.subtract(y_hatA, y_hatB), 0.), tf_penalty)
    loss = weight * tf.divide(tf.reduce_sum(tf.square(tf_error)), data.forwardpass_counts[labelBase+'_x'])
    # --- ORIGINAL
    # loss = weight * tf.divide(tf.reduce_sum(tf.square(tf.math.maximum(tf.subtract(y_hatA, y_hatB), 0.))),
    #                           data.forwardpass_counts[labelBase+'_x'])
    return loss
