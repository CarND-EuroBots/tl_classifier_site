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

## Train

## Evaluate

## Freeze inference graph
