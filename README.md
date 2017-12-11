# tl_classifier_site
Traffic Light classifier for the site scenario

## Annotate data from a rosbag
Required: to have ROS installed

    $ ./annotate.py <path_to_input_rosbag> <path_to_output_directory>

**Instructions**

1. Left click on top-left corner of bounding box, and **keep pressing**
2. Drag mouse to bottom-down corner and **release mouse**. A message
displaying the selected coordinates should be displayed
3. Press 'g', 'y', or 'r' keys to select the tag of the image: 'Green', 'Yellow'
or 'Red', respectively. Press 's' to skip image or 'q' for quitting.

## Create TFrecord

    $ docker run --rm=true -it -v=$(pwd):$(pwd) -w=$(pwd) -u=$(id -u):$(id -g) carlosgalvezp/tf-object-detection:latest /bin/bash

    $ ./create_udacity_tf_record.py -i data/just_traffic_light_image_proc/just_traffic_light_image_proc/just_traffic_light_image_proc_annotations.csv -o data/just_traffic_light_image_proc/just_traffic_light_image_proc/ -l traffic_light_label_map.pbtxt 


## Train

nvidia-docker run --rm=true -it -v=$(pwd):$(pwd) -w=$(pwd) -u=$(id -u):$(id -g) carlosgalvezp/tf-object-detection:latest /bin/bash

python /opt/models/research/object_detection/train.py --logtostderr --pipeline_config_path=/home/cgalvezd/git/tl_classifier_site/ssd_mobilenet_v1_site.config --train_dir=/home/cgalvezd/git/tl_classifier_site/train_sandbox/models/ssd_mobilenet_v1_coco_2017_11_17/train

## Evaluate

export CUDA_VISIBLE_DEVICES=

python /opt/models/research/object_detection/eval.py --logtostderr --pipeline_config_path=/home/cgalvezd/git/tl_classifier_site/ssd_mobilenet_v1_site.config --checkpoint_dir=/home/cgalvezd/git/tl_classifier_site/train_sandbox/models/ssd_mobilenet_v1_coco_2017_11_17/train --eval_dir=/home/cgalvezd/git/tl_classifier_site/train_sandbox/models/ssd_mobilenet_v1_coco_2017_11_17/eval

## Tensorboard
tensorboard --logdir=/home/cgalvezd/git/tl_classifier_site/train_sandbox/models/ssd_mobilenet_v1_coco_2017_11_17/

## Freeze inference graph

python /opt/models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path ssd_mobilenet_v1_site.config --trained_checkpoint_prefix train_sandbox/models/ssd_mobilenet_v1_coco_2017_11_17/output_model/model.ckpt-83267 --output_directory train_sandbox/models/ssd_mobilenet_v1_coco_2017_11_17/output_model
