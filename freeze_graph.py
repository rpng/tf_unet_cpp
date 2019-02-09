#!/usr/bin/python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from google.protobuf import text_format

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework.errors_impl import NotFoundError

try:
    from tensorflow.contrib.nccl.python.ops import nccl_ops
    nccl_ops._maybe_load_nccl_ops_so()
except NotFoundError:
    pass # only cpu or only single gpu so no nccl installed
except AttributeError:
    pass # only cpu or only single gpu so no nccl installed
except ImportError:
    pass # only cpu or only single gpu so no nccl installed

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph 

dir = os.path.dirname(os.path.realpath(__file__))

def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 

    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
 
    print("\n\nLoading checkpoint: %s\n\n" % input_checkpoint)

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    if not os.path.isdir('frozen_graphs'):
        os.mkdir('frozen_graphs')
    output_graph = "frozen_graphs/unet_frozen.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)
        
        gd = tf.get_default_graph().as_graph_def()

        """
        # fix batch norm nodes
        for node in gd.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in xrange(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
        """

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            gd, # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def

def optimize_for_inference():

    input_graph_def = graph_pb2.GraphDef()
    inpt = "frozen_graphs/unet_frozen.pb"
    output = "cpp/unet.pb"
    with gfile.Open(inpt, "rb") as f:
        data = f.read()
        #if FLAGS.frozen_graph:
        input_graph_def.ParseFromString(data)
        #else:
        #    text_format.Merge(data.decode("utf-8"), input_graph_def)
    
    input_names = ["UNet/images"]
    output_names = ["UNet/mask"]

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
      input_graph_def,
      input_names,
      output_names,
      dtypes.float32.as_datatype_enum,
      False)
    f = gfile.FastGFile(output, "w")
    f.write(output_graph_def.SerializeToString())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="model", help="Model folder to export")
    args = parser.parse_args()

    freeze_graph(args.model_dir, "UNet/mask")
    print("Graph frozen successfully, optimizing for inference...")
    optimize_for_inference()
    print("Optimized successfully, done.")
