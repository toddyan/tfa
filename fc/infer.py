import tensorflow as tf

def get_regularized_weight(name, shape, regul):
    w = tf.get_variable(name, shape=shape, dtype=tf.float32)
    if regul != None:
        tf.add_to_collection("losses",regul(w))
    return w
def infer(x, fc_layers, regul):
    a = x
    for layer_id in range(1,len(fc_layers)):
        layer_name = "layer" + str(layer_id)
        with tf.variable_scope(layer_name, initializer=tf.truncated_normal_initializer(stddev=0.1)):
            w = get_regularized_weight("weight", fc_layers[layer_id-1:layer_id+1], regul)
            b = tf.get_variable("bias", shape=fc_layers[layer_id])
            z = tf.matmul(a, w) + b
            if layer_id == len(fc_layers)-1:
                return z
            a = tf.nn.relu(z)

