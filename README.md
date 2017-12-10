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

    $ docker run --rm=true -it -v=$(pwd):$(pwd) -w=$(pwd) -u=$(id -u):$(id -g) carlosgalvezp/tf-object-detection /bin/bash

    $ ./create_udacity_tf_record.py -i data/just_traffic_light_image_proc/just_traffic_light_image_proc/just_traffic_light_image_proc_annotations.csv -o data/just_traffic_light_image_proc/just_traffic_light_image_proc/ -l traffic_light_label_map.pbtxt 


## Train

## Evaluate

## Freeze inference graph
