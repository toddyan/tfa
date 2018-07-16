import globalconf
import os
import tensorflow as tf
import numpy as np
import glob
from tensorflow.python.platform import gfile

def create_image_lists(s, input_dir, valid_rate, test_rate):
    sub_dirs = [x[0] for x in os.walk(input_dir)][1:]
    extensions = ['jpg','jepg','JPG','JPEG']

    training_images = []
    training_labels = []
    validation_images = []
    validation_labels = []
    testing_iamges = []
    testing_labels = []
    label = 0
    for dir in sub_dirs:
        print(dir)
        files = []
        dir_name = os.path.basename(dir)
        for ext in extensions:
            file_glob = os.path.join(input_dir, dir_name, '*.' + ext)
            files.extend(glob.glob(file_glob))
        for file in files:
            image_data = gfile.FastGFile(file, 'rb').read()
            image = tf.image.decode_jpeg(image_data)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize_images(image, (299,299))
            value = s.run(image)
            rand = np.random.rand()
            if rand <= test_rate:
                testing_iamges.append(value)
                testing_labels.append(label)
            elif rand <= test_rate + valid_rate:
                validation_images.append(value)
                validation_labels.append(label)
            else:
                training_images.append(value)
                training_labels.append(label)
        label += 1
    random_state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(random_state)
    np.random.shuffle(training_labels)
    return np.asarray([training_images,training_labels,validation_images,validation_labels,testing_iamges,testing_labels])
def main(argv=None):
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    input_dir = globalconf.get_root() + 'transfer/flower_photos'
    output_path = globalconf.get_root() + 'transfer/flower_photos/flower_photos.npy'
    valid_rate = 0.1
    test_rate = 0.1
    with tf.Session() as s:
        data = create_image_lists(s, input_dir, valid_rate, test_rate)
        print(data.shape)
        np.save(output_path, data)

if __name__ == "__main__":
    tf.app.run()