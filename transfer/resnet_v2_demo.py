import globalconf
from inception_resnet_v2 import inception_resnet_v2_arg_scope,inception_resnet_v2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc as misc
import PIL.Image as Image


# http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
checkpoint_path = globalconf.get_root() + "transfer/resnet_v2/inception_resnet_v2_2016_08_30.ckpt"
logdir = globalconf.get_root() + "transfer/resnet_v2/logdir"
height, width, channels = 299, 299, 3
x = tf.placeholder(tf.float32, shape=[None, height, width, channels])
arg_scope = inception_resnet_v2_arg_scope()
for k,v in arg_scope.items():
    print(k,":")
    for k2,v2 in v.items():
        print("\t",k2,":",v2)
with slim.arg_scope(arg_scope):
    logits, end_points = inception_resnet_v2(x, is_training=False, num_classes=1001)
    features = end_points['PreLogitsFlatten']
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

#writer = tf.summary.FileWriter(logdir=logdir, graph=tf.get_default_graph())
#writer.close()


img = Image.open("D:/tfroot/transfer/resnet_v2/sample/0.jpg").convert('RGB').resize((299,299),Image.ANTIALIAS)
metrix = np.asarray([misc.fromimage(img)])
fea = sess.run(features, feed_dict={x:metrix})
print(fea.shape == (1,1536))
