import numpy as np
from keras.datasets import mnist
import platform
class Conf:
    def __init__(self):
        ############### path ################
        if platform.system() == "Windows":
            self.root="D:/tfroot/"
        elif platform.system() == "Linux":
            self.root="/home/yxd/tfroot/"
        elif platform.system() == "Darwin":
            self.root="/Users/yxd/tfroot/"
        self.model_dir = self.root + "fc/mnist/"
        self.model_name = "mnist.ckpt"
        self.model_path = self.model_dir + self.model_name
        ############### data ################
        (x_train,y_train),(x_test,y_test) = mnist.load_data()
        validation_example = int(0.1 * x_train.shape[0])
        self.x_train = x_train.reshape(x_train.shape[0],-1)/255.0
        self.x_valid = self.x_train[validation_example:]
        self.x_train = self.x_train[:validation_example]
        self.x_test = x_test.reshape(x_test.shape[0],-1)/255.0
        self.y_train = y_train.astype(np.int32)
        self.y_valid = self.y_train[validation_example:]
        self.y_train = self.y_train[:validation_example]
        self.y_test = y_test.astype(np.int32)

        self.N = [self.x_train.shape[1],500,10]
        self.batch_size = 6400
        self.cursor = 0
        ############### optimize ################

    def get_root(self):
        return self.root
    def get_batch(self):
        if self.cursor >= self.x_train.shape[0]:
            self.cursor = 0
        start = self.cursor
        end = min(self.cursor+self.batch_size, self.x_train.shape[0])
        self.cursor = end
        return self.x_train[start:end], self.y_train[start:end]
