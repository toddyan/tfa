import globalconf
from inception_resnet_v2 import inception_resnet_v2_arg_scope,inception_resnet_v2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc as misc
import PIL.Image as Image

class ResNetV2FeatureExtractor:
    def __init__(self, checkpoint_path):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.height, self.width, self.channels = 299, 299, 3
            self.x = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channels])
            arg_scope = inception_resnet_v2_arg_scope()
            with slim.arg_scope(arg_scope):
                logits, end_points = inception_resnet_v2(self.x, is_training=False, num_classes=1001)
                self.features = end_points['PreLogitsFlatten']
                saver = tf.train.Saver()
                saver.restore(self.sess, checkpoint_path)

    def get_feature(self, input_x):
        with self.graph.as_default():
            fea = self.sess.run(self.features, feed_dict={self.x: input_x})
            return fea


checkpoint_path = globalconf.get_root() + "transfer/resnet_v2/inception_resnet_v2_2016_08_30.ckpt"
resnet = ResNetV2FeatureExtractor(checkpoint_path)

img = Image.open("D:/tfroot/transfer/resnet_v2/sample/0.jpg").convert('RGB').resize((299,299),Image.ANTIALIAS)
metrix = np.asarray([misc.fromimage(img)])

fea = resnet.get_feature(metrix)
print(fea.shape == (1,1536))
print(','.join([str(e) for e in fea[0]]))