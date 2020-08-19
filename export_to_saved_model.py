#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import horovod.tensorflow as hvd  # Necessary to register Horovod OPs that may exists

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.saved_model import tag_constants
from tensorflow.core.protobuf import saver_pb2

PATH_TO_CHECKPOINT = "./results"

print("\nListing files in the `{}` directory ...".format(PATH_TO_CHECKPOINT))
for file in os.listdir(PATH_TO_CHECKPOINT):
    print("\t{}".format(file))

print("\nLet's freeze the training graph for inference ...")

model_name = "resnet50v1.5"
input_graph_path = os.path.join(PATH_TO_CHECKPOINT, 'graph.pbtxt')
input_checkpoint = tf.train.latest_checkpoint(PATH_TO_CHECKPOINT)
input_saver_def_path = ""
input_binary = False
output_node_names = "resnet50/output/softmax,resnet50/output/dense/BiasAdd"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = os.path.join(
    PATH_TO_CHECKPOINT, 'frozen_{}.pb'.format(model_name)
)
clear_devices = True

print("Loading Graph `{}` ...".format(input_graph_path))
print("Loading Checkpoint `{}` ...".format(input_checkpoint))
# https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/python/tools/freeze_graph.py#L286-L301
freeze_graph.freeze_graph(
    input_graph=input_graph_path,  # A `GraphDef` file to load.
    input_saver=input_saver_def_path,  #  A TensorFlow Saver file.
    input_binary=input_binary,  #  A Bool. True means input_graph is .pb, False indicates .pbtxt.
    input_checkpoint=input_checkpoint,
    output_node_names=output_node_names,  # The name(s) of the output nodes, comma separated.
    restore_op_name="",  # Unused
    filename_tensor_name="",  # Unused
    output_graph=output_frozen_graph_name,  # String where to write the frozen `GraphDef`.
    clear_devices=clear_devices,  #  A Bool whether to remove device specifications.
    initializer_nodes="",  # Comma separated list of initializer nodes to run before freezing.
    variable_names_whitelist="",  # The set of variable names to convert (optional, bydefault, all variables are converted)
    variable_names_blacklist="",  # The set of variable names to omit converting to constants (optional)
    input_meta_graph=None,  # A `MetaGraphDef` file to load (optional).
    input_saved_model_dir=None,  # Path to the dir with TensorFlow 'SavedModel' file and variables (optional).
    saved_model_tags=tag_constants.SERVING,  # Group of comma separated tag(s) of the MetaGraphDef to load, in string format.
    checkpoint_version=saver_pb2.SaverDef.V2  # Tensorflow variable file format (saver_pb2.SaverDef.V1or saver_pb2.SaverDef.V2
)

print("\nLet's optimize the frozen graph for inference ...")

frozen_graph_def = tf.compat.v1.GraphDef()
with tf.io.gfile.GFile(output_frozen_graph_name, "rb") as f:
    data = f.read()
    frozen_graph_def.ParseFromString(data)

optimized_frozen_graph = optimize_for_inference_lib.optimize_for_inference(
        frozen_graph_def,
        ["StagingArea_1_get"], # an array of the input node(s)
        ["resnet50/output/softmax", "resnet50/output/dense/BiasAdd"], # an array of output nodes
        tf.float32.as_datatype_enum)

output_frozen_graph_name = os.path.join(
    PATH_TO_CHECKPOINT, 'optimized_frozen_{}.pb'.format(model_name)
)
with tf.io.gfile.GFile(output_frozen_graph_name, "w") as f:
    f.write(optimized_frozen_graph.SerializeToString())

print('\nThe optimized TensorFlow SavedModel has been saved: `{}` ...'.format(
      output_frozen_graph_name
))
