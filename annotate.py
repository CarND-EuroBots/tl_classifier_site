#!/usr/bin/env python
""" Utility program to read a ROS bag, extract images from it and annotate
    the bounding box of the traffic light """
import os
import argparse
import rosbag
import cv2
from cv_bridge import CvBridge

def help():
    output = \
    """Annotate traffic lights from a rosbag.
    Instructions:
    1) Click on the top-left and bottom-right corners of the traffic light
    2) Type the category of the light:
        2.1 - 'g' for green
        2.2 - 'r' for red
        2.3 - 'y' for yellow
        2.4 - 's' to skip image
    3) Press 'q' to quit before covering the whole dataset
    """

    return output

def get_arguments():
    parser = argparse.ArgumentParser(description='Annotate traffic lights from a rosbag')
    parser.add_argument('rosbag', help='input .bag file')
    parser.add_argument('output', help='output folder')
    parser.add_argument('--append', action='store_true', help='if true, append to existing CSV file')
    parser.add_argument('--start', type=int, help='starting message number', default=0)

    return parser.parse_args()

xy_top_left = []
xy_bottom_right = []

def mouse_callback(event, x, y, flags, param):
    global xy_top_left, xy_bottom_right

    if event == cv2.EVENT_LBUTTONDOWN:
        xy_top_left = [x, y]
        print('Registered top-left corner: {}'.format(xy_top_left))
    elif event == cv2.EVENT_LBUTTONUP:
        xy_bottom_right = [x, y]
        print('Registered bottom-right corner: {}'.format(xy_bottom_right))
        print('Press r for red, g for green, y for yellow or s to skip')

def annotate_bag(input_bag_path, output_folder, append, msg_start):
    # Create bag object ang get number of messages
    bag = rosbag.Bag(input_bag_path)
    image_topic = '/image_color'
    n_messages = bag.get_message_count(topic_filters=image_topic)

    # Create ROS-OpenCV bridge
    cv_bridge = CvBridge()

    # Create OpenCV window and set click callback
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback)

    # Define output name
    input_name, _ = os.path.splitext(os.path.basename(input_bag_path))
    output_folder = os.path.join(output_folder, input_name)
    output_name = os.path.join(output_folder, '{}_annotations.csv'.format(input_name))

    # Create output folder if it doesn't exist
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    else:
        if not append:
            raise RuntimeError('The folder {} already exists! Use --apend or '
                               'delete it first'.format(output_folder))

    # Open output filename
    write_type = 'a' if os.path.isfile(output_name) and append else 'w'
    with open(output_name, write_type) as output_file:
        msg_number = 0
        # Loop over all images in the rosbag
        for _, msg, _ in bag.read_messages(topics=image_topic):
            if msg_number >= msg_start:
                print('Message {}/{}...'.format(msg_number, n_messages))

                # Convert from ROS to CV image
                # Use BGR so that when calling cv.imwrite it will write the
                # channels in the correct order. When training, the image
                # will be read by PIL, which will read RGB, same as in the
                # runtime code, where we get the OpenCV image as rgb8 encoded
                cv_image = cv_bridge.imgmsg_to_cv2(msg, "bgr8")

                # Display image
                cv2.imshow('Image', cv_image)

                # Wait for user to provide the annotation
                is_valid_key = False
                exit_loop = False
                while not is_valid_key:
                    key = cv2.waitKey(0) & 0xFF

                    is_valid_key = True
                    if key == ord("g"):
                        class_id = 'Green'
                    elif key == ord("y"):
                        class_id = 'Yellow'
                    elif key == ord("r"):
                        class_id = 'Red'
                    elif key == ord("s"):
                        class_id = 'Unknown'
                    elif key == ord("q"):
                        exit_loop = True
                    else:
                        is_valid_key = False
                        print('Invalid key {}, try again!'.format(chr(key)))

                if exit_loop:
                    break

                if class_id != 'Unknown':
                    # Save image to disk
                    img_fname = '{}_{}.jpg'.format(input_name, msg_number)
                    img_path = os.path.join(output_folder, img_fname)
                    print(img_path)
                    cv2.imwrite(img_path, cv_image)

                    # Compute normalized coordinates of the bounding box
                    height = cv_image.shape[0]
                    width = cv_image.shape[1]

                    x1n = float(xy_top_left[0]) / float(width)
                    y1n = float(xy_top_left[1]) / float(height)

                    x2n = float(xy_bottom_right[0]) / float(width)
                    y2n = float(xy_bottom_right[1]) / float(height)

                    assert x1n >= 0.0 and x1n <= 1.0
                    assert y1n >= 0.0 and y1n <= 1.0
                    assert x2n >= 0.0 and x2n <= 1.0
                    assert y2n >= 0.0 and y2n <= 1.0

                    # Write to CSV file
                    output_file.write('{}, {}, {}, {}, {}, {}\n'.format(img_path,
                                                                        class_id,
                                                                        x1n, y1n,
                                                                        x2n, y2n))
            msg_number += 1
    # Stop application
    cv2.destroyAllWindows()

def main():
    print(help())
    args = get_arguments()
    annotate_bag(os.path.abspath(args.rosbag),
                 os.path.abspath(args.output),
                 args.append, args.start)

if __name__ == "__main__":
    main()
