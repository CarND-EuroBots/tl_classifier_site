# Traffic Light Classifier - Site scenario

This repository contains the data and training scripts for creating the
frozen graph to be used in the CarND-Capstone repository, for running
traffic light detection on Carla.

## Run session inside the docker image
To ensure all the dependencies are installed, run an interactive docker session:

    nvidia-docker run --rm=true -it -v=$(pwd):$(pwd) -w=$(pwd) -u=$(id -u):$(id -g) carlosgalvezp/tf-object-detection:latest /bin/bash

Mount additional volumes as required.

## Annotate data from a rosbag
The first thing to do when training is to have a dataset.
Either use the provided `data` can be used, or new images can be obtained
from a rosbag as follows:


    $ ./annotate.py <path_to_input_rosbag> <path_to_output_directory>


**Instructions**

1. Left click on top-left corner of bounding box, and **keep pressing**
2. Drag mouse to bottom-down corner and **release mouse**. A message
displaying the selected coordinates should be displayed.
3. Press 'g', 'y', or 'r' keys to select the tag of the image: 'Green', 'Yellow'
or 'Red', respectively. Press 's' to skip image or 'q' for quitting.

## Create TFrecord

Once the dataset has been generated, a `TFrecord` needs to be generated
from it to plug into TensorFlow's Object Detection API


    $ ./create_udacity_tf_record.py -i <input_annotations.csv> -o <output_path> -l traffic_light_label_map.pbtxt 


The output path must a file with `.record` extension.

You should create 2 datasets:

- `train.record`
- `val.record`

## Train


For training, follow these steps:

1. Create a sandbox folder:

       mkdir sandbox

2. Download pre-trained model from the [zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
[Faster RCNN](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz) was used for the project.

3. Decompress the model in the sandbox.

4. Create a `data` directory and put the previous 2 datasets inside.

5. Download and configure the pipeline configuration file from [here](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs).
   Mostly the only required change is to choose the paths to the model checkpoint,
   the train and validation datasets (`.record`).

6. The folder structure should look like this:

   ```
   ├── sandbox
   │   └── faster_rcnn_resnet101_coco_11_06_2017
   │       ├── data
   │       │   ├── train.record
   │       │   └── val.record
   │       ├── frozen_inference_graph.pb
   │       ├── graph.pbtxt
   │       ├── model.ckpt.data-00000-of-00001
   │       ├── model.ckpt.index
   │       ├── model.ckpt.meta
   │       ├── pipeline.config
   │       └── traffic_light_label_map.pbtxt
   ```

7. Run training. For example:

       python /opt/models/research/object_detection/train.py --logtostderr --pipeline_config_path=<path_to_pipeline.config> --train_dir=sandbox/<model>/train

   The results from training will be stored in the `sandbox/<model>/train` folder.
   The training will now start and will run indefinitely until the user interrupts the process.
   Monitor the performance on the validation dataset and stop the training when the results
   are good.
  

## Evaluate

On a separate terminal, evaluation can be run in parallel to observe the networks'
performance over time and know when to stop training. Open another docker session and run:

    export CUDA_VISIBLE_DEVICES=
    python /opt/models/research/object_detection/eval.py --logtostderr --pipeline_config_path=<path_to_pipeline.config> --checkpoint_dir=sandbox/<model>/train --eval_dir=sandbox/<model>/eval

## Tensorboard

On a separate terminal, the training and evaluation progress can be visually monitored using TensorBoard:

    tensorboard --logdir=sandbox/<model>

## Freeze inference graph

Once the model is trained, copy the files `model.ckpt-<number>*` to a folder `output_model`.
Then, freeze the model running:

    python /opt/models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path <path_to_pipeline.config> --trained_checkpoint_prefix sandbox/<model>/output_model/model.ckpt-<number> --output_directory sandbox/<model>/output_model/model.ckpt-<number>

The result is a `frozen_inference_graph.pb` file that can be loaded directly into TensorFlow.

## Split model into chunks

Optionally, the model can be splitted into chunks to be able to store them in Git
without the Git LFS requirement. To do so, run the following,
**from the CarND-Capstone** folder:

    python ros/src/tl_detector/light_classification model_tools.py <path_to_frozen_inference_graph.pb>

It will create a folder `frozen_inferenace_graph.pb_chunks` and put the chunks inside.

    
