import globalconf
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile

def npy_maker(input_dir):
    inputs = []
    for d in os.listdir(input_dir):
        if not os.path.isdir(os.path.join(input_dir, d)): continue
        inputs.append([len(inputs), os.path.join(input_dir, d)])
    # class by class process
    for task in inputs:
        print(task)
        with tf.Session() as s:
            part_npy_path = os.path.join(input_dir, str(task[0]) + ".npy")
            arr = []
            for img in os.listdir(task[1]):
                if not img.endswith(".jpg"): continue
                file = os.path.join(task[1], img)
                image_data = gfile.FastGFile(file, 'rb').read()
                image = tf.image.decode_jpeg(image_data)
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                image = tf.image.resize_images(image, (299, 299))
                value = s.run(image)
                arr.append(value)
            nparr = np.asarray(arr)
            np.save(part_npy_path, nparr)
            del arr
            del nparr
def combine(input_dir,num_classes,valid_rate,test_rate):
    training_images = []
    training_labels = []
    validation_images = []
    validation_labels = []
    testing_iamges = []
    testing_labels = []
    for label in range(num_classes):
        part_npy_path = os.path.join(input_dir, str(label) + ".npy")
        nparr = np.load(part_npy_path)
        print(nparr.shape)
        for img in nparr:
            rand = np.random.rand()
            if rand <= test_rate:
                testing_iamges.append(img)
                testing_labels.append(label)
            elif rand <= test_rate + valid_rate:
                validation_images.append(img)
                validation_labels.append(label)
            else:
                training_images.append(img)
                training_labels.append(label)
    random_state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(random_state)
    np.random.shuffle(training_labels)
    return np.asarray(training_images), np.asarray(training_labels),\
           np.asarray(validation_images), np.asarray(validation_labels),\
           np.asarray(testing_iamges), np.asarray(testing_labels)
def main(argv=None):
    img_train, label_train, img_valid, label_valid, img_test, label_test\
        = combine(globalconf.get_root() + 'transfer/small', 5, 0.1,0.1)
    print(img_train.shape, label_train.shape, img_valid.shape, label_valid.shape, img_test.shape, label_test.shape)
if __name__ == "__main__":
    tf.app.run()