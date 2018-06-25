import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as cont_layers
from fc import mnist_conf

conf = mnist_conf.Conf()

r = cont_layers.l2_regularizer(0.0001)

def get_regularized_weight(shape, regul):
    pass
def infer(x):
    with tf.variable_scope("layer1"):
        pass