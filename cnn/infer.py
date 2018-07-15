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

def cnn_infer(x, regul):
    initializer = tf.truncated_normal_initializer(stddev=0.1)
    with tf.variable_scope("layer1",initializer=initializer):
        w = tf.get_variable(
            "filters",
            shape=[5,5,1,16],
            dtype=tf.float32
        )
        b = tf.get_variable(
            "bias",
            shape=[16],
            dtype=tf.float32
        )
        a = tf.nn.conv2d(x, w, strides=[1,1,1,1],padding='SAME')
        a = tf.nn.relu(tf.nn.bias_add(a, b)) # m*28*28*8
        a = tf.nn.max_pool(a, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')# m*14*14*8
    with tf.variable_scope("layer2", initializer=initializer):
        w = tf.get_variable(
            "filters",
            shape=[5, 5, 16, 64],
            dtype=tf.float32
        )
        b = tf.get_variable(
            "bias",
            shape=[64],
            dtype=tf.float32
        )
        a = tf.nn.conv2d(a, w, strides=[1, 1, 1, 1], padding='SAME')
        a = tf.nn.relu(tf.nn.bias_add(a, b))  # m*14*14*16
        a = tf.nn.max_pool(a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # m*7*7*16
    cnn_out_size = a.get_shape().as_list()
    flatten_size = cnn_out_size[1]*cnn_out_size[2]*cnn_out_size[3]
    flatten = tf.reshape(a, [-1,flatten_size])

    with tf.variable_scope("layer3", initializer=initializer):
        w = get_regularized_weight("weight", [flatten_size, 512], regul)
        b = tf.get_variable("bias", shape=[512])
        z = tf.matmul(flatten, w) + b
        a = tf.nn.relu(z)
    with tf.variable_scope("layer4", initializer=initializer):
        w = get_regularized_weight("weight", [512, 10], regul)
        b = tf.get_variable("bias", shape=[10])
        z = tf.matmul(a, w) + b
        return z