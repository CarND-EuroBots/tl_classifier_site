#!/usr/bin/env python
"""Example usage:
    python create_udacity_tf_record.py \
        --annotations_path=/home/user/udacity/data/annotations.csv \
        --output_path=/home/user/udacity/output.record
        --label_map_path=/home/user/label_map.pbtxt
"""
import io
import os
import random
import argparse
import PIL.Image
import tensorflow as tf
import numpy as np

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


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

def create_tf_record(examples, output_path, label_map_dict):
    writer = tf.python_io.TFRecordWriter(output_path)

    for example in examples:
        tf_example = create_tf_example(example, label_map_dict)
        writer.write(tf_example.SerializeToString())

    writer.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Create TF record')

    parser.add_argument('-i', dest='annotations_paths', action='append')
    parser.add_argument('-o', dest='output_path')
    parser.add_argument('-l', dest='label_map_path')

    return parser.parse_args()

def main():
    examples = []

    # Read CSV files and store information into "examples"
    args = parse_arguments()
    annotations_files = args.annotations_paths

    for annotations_file in annotations_files:
        print('Processing annotatations from {}...'.format(annotations_file))
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

    # Shuffle to avoid processing sequences
    random.seed(42)
    random.shuffle(examples)

    # Create TF records
    label_map_dict = label_map_util.get_label_map_dict(args.label_map_path)
    create_tf_record(examples, args.output_path, label_map_dict)

if __name__ == '__main__':
    main()
