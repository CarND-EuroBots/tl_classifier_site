#!/usr/bin/env python3
"""Example usage:
    python create_udacity_tf_record.py \
        --annotations_path=/home/user/udacity/data/annotations.csv \
        --output_path=/home/user/udacity/output.record
        --label_map_path=/home/user/label_map.pbtxt
"""
import io
import os
import random
import tensorflow as tf
import PIL.Image
import numpy as np

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('annotations_path', '', 'Path to input data')
flags.DEFINE_string('output_dir', '', 'Directory to store output TFRecords')
flags.DEFINE_string('label_map_path', '', 'Path to label map proto')
FLAGS = flags.FLAGS


def create_tf_example(example, label_map_dict):
    # Read raw encoded bytes from image
    with tf.gfile.GFile(example['filename'], 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')

    width = image.size[0] # Image width
    height = image.size[1] # Image height

    filename = example['filename'] # Filename of the image. Empty if image is not from file
    encoded_image_data = encoded_jpg # Encoded image bytes
    image_format = 'jpeg'

    xmins = [example['x1n']] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [example['x2n']] # List of normalized right x coordinates in bounding box (1 per box)
    ymins = [example['y1n']] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [example['y2n']] # List of normalized bottom y coordinates in bounding box (1 per box)

    classes = [int(label_map_dict[example['class_id']])] # List of string class name of bounding box (1 per box)
    classes_text = [example['class_id'].encode('utf8')] # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def create_tf_record(examples, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    for example in examples:
        tf_example = create_tf_example(example, label_map_dict)
        writer.write(tf_example.SerializeToString())
    writer.close()


def main(_):
    # Read CSV and store information into "examples"
    examples = []
    annotations_file = FLAGS.annotations_path

    with open(annotations_file, 'r') as fid:
        for line in fid:
            csv_data = line.split(', ')
            example = {'filename': csv_data[0],
                       'class_id': csv_data[1],
                       'x1n': float(csv_data[2]),
                       'y1n': float(csv_data[3]),
                       'x2n': float(csv_data[4]),
                       'y2n': float(csv_data[5])}

            examples.append(example)

    # Shuffle
    random.seed(42)
    random.shuffle(examples)

    # Split into training and validation sets
    num_examples = len(examples)
    num_train = int(0.7 * num_examples)

    train_examples = examples[:num_train]
    val_examples = examples[num_train:]

    train_output_path = os.path.join(FLAGS.output_dir, 'train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'val.record')

    # Create TF records
    create_tf_record(train_examples, train_output_path)
    create_tf_record(val_examples, val_output_path)

if __name__ == '__main__':
    tf.app.run()
