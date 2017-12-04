#!/usr/bin/env python3
"""Example usage:
    python create_udacity_tf_record.py \
        --annotations_path=/home/user/udacity/data/annotations.csv \
        --output_path=/home/user/udacity/output.record
        --label_map_path=/home/user/label_map.pbtxt
"""
import io
import os
import tensorflow as tf
import PIL.Image
import numpy as np

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('annotations_path', '', 'Path to input data')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
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


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    # Read CSV
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

            # Append example to TF record
            tf_example = create_tf_example(example, label_map_dict)
            writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
