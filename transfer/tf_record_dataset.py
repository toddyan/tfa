import tensorflow as tf
import os
import numpy as np
import globalconf
import random
import PIL.Image as Image


class ImageTFRecordBuilder:
    def __init__(self,height,width,channel,piece_size,src_root,dst_root,flower_id_map,valid_ratio,test_ratio):
        self.height = height
        self.width = width
        self.channel = channel
        self.piece_size = piece_size
        self.src_root = src_root
        self.dst_root = dst_root
        self.flower_id_map = flower_id_map
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio

    def make_tfrecord(self):
        class WriterWrapper:
            def __init__(self, dst_root, piece_size):
                self.train_count = 0
                self.valid_count = 0
                self.test_count = 0
                self.dst_root = dst_root
                self.piece_size = piece_size
                self.train_writer = None
                self.valid_writer = None
                self.test_writer = None

            def get_train_writer(self):
                if self.train_count % self.piece_size == 0:
                    if self.train_count != 0: self.train_writer.close()
                    piece_id = str(int(self.train_count / self.piece_size))
                    self.train_writer = tf.python_io.TFRecordWriter(
                        os.path.join(self.dst_root, "train-" + piece_id)
                    )
                self.train_count += 1
                return self.train_writer

            def get_valid_writer(self):
                if self.valid_count % self.piece_size == 0:
                    if self.valid_count != 0: self.valid_writer.close()
                    piece_id = str(int(self.valid_count / self.piece_size))
                    self.valid_writer = tf.python_io.TFRecordWriter(
                        os.path.join(self.dst_root, "valid-" + piece_id)
                    )
                self.valid_count += 1
                return self.valid_writer

            def get_test_writer(self):
                if self.test_count % self.piece_size == 0:
                    if self.test_count != 0: self.test_writer.close()
                    piece_id = str(int(self.test_count / self.piece_size))
                    self.test_writer = tf.python_io.TFRecordWriter(
                        os.path.join(self.dst_root, "test-" + piece_id)
                    )
                self.test_count += 1
                return self.test_writer

            def close(self):
                if self.train_writer is not None: self.train_writer.close()
                if self.valid_writer is not None: self.valid_writer.close()
                if self.test_writer is not None: self.test_writer.close()
        img_id_list = []
        for flower_name, class_id in self.flower_id_map.items():
            img_dir = os.path.join(self.src_root,flower_name)
            for f in os.listdir(img_dir):
                if not f.endswith(".jpg"): continue
                img_path = os.path.join(img_dir, f)
                img_id_list.append((img_path, class_id))
        random.shuffle(img_id_list)
        random.shuffle(img_id_list)
        writer_wrapper = WriterWrapper(self.dst_root, self.piece_size)
        _log_count = 0
        for img, id in img_id_list:
            rand = np.random.rand()
            if rand < self.test_ratio:
                writer = writer_wrapper.get_test_writer()
            elif rand < self.test_ratio + self.valid_ratio:
                writer = writer_wrapper.get_valid_writer()
            else:
                writer = writer_wrapper.get_train_writer()
            npy_img = np.array(
                Image.open(img).convert('RGB').resize((self.height, self.width), Image.ANTIALIAS)
            ).astype(np.uint8)
            example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[npy_img.tostring()])),
                "label":     tf.train.Feature(int64_list=tf.train.Int64List(value=[id]))
            }))
            writer.write(example.SerializeToString())
            _log_count += 1
            print(_log_count, "of", len(img_id_list))
        writer_wrapper.close()
    def get_parser(self):
        def parser(record):
            features = tf.parse_single_example(record, features={
                "image": tf.FixedLenFeature([], tf.string),
                "label": tf.FixedLenFeature([], tf.int64)
            })
            image = tf.reshape(tf.decode_raw(features["image"],tf.uint8), [self.height, self.width, self.channel])
            image = tf.image.convert_image_dtype(image, tf.float32)
            label = features["label"]
            return image, label
        return parser


if __name__ == "__main__1":
    src_root = globalconf.get_root() + "transfer/small/"
    dst_root = globalconf.get_root() + "transfer/tfrecord/"
    flower_id_map = {"daisy":0,"dandelion":1,"roses":2,"sunflowers":3,"tulips":4}
    # res_a = tf.reshape(tf.decode_raw(a.tostring(),tf.uint8), [h,w,c])
    builder = ImageTFRecordBuilder(299,299,3,128,src_root,dst_root,flower_id_map,0.1,0.1)
    builder.make_tfrecord()

if __name__ == "__main__":
    dst_root = globalconf.get_root() + "transfer/tfrecord/"
    files = tf.train.match_filenames_once(dst_root + "valid-*")
    paser = ImageTFRecordBuilder(299, 299, 3, None, None, None, None, None, None).get_parser()
    ds = tf.data.TFRecordDataset(files)
    ds = ds.map(paser).shuffle(buffer_size=100).batch(32)
    iterator = ds.make_initializable_iterator()
    image, label = iterator.get_next()
    with tf.Session() as s:
        s.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        s.run(iterator.initializer)
        img, lab = s.run([image, label])
        print(img, lab)