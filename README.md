# How to train with Tensorflow 1.x and perform inference with Tensorflow 2.x using TF-TRT ?

**NOTE: WE SHALL PROBABLY EXPLAIN WHY WE WANT PEOPLE TO USE TF2.x. Long story short: TF1.x TF-TRT is not updated any more, a number of optimizations and new features are not available in TF1.x (e.g. Dynamic Shape, new layers are supported, etc.)**.

This blogpost will focus on explaining how to to use a model trained with TF1.x and execute inference in TF2.x. In this sense, we will use [Resnet50v1.5](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5) published as part of the [NVIDIA Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples) github repository. This process can be extrapolated to any other models.

In this blogpost we will use the latest available tensorflow containers (at the time of writing):

* **Tensorflow 1.x Container**: `docker pull nvcr.io/nvidia/tensorflow:20.07-tf1-py3`
* **Tensorflow 2.x Container**: `docker pull nvcr.io/nvidia/tensorflow:20.07-tf2-py3`

Prerequisites to be able to follow this blog posts/tutorial:
- We currently only support Ubuntu, you can try with other systems however we can't guarantee success.

- Docker: https://docs.docker.com/get-docker/

- NVIDIA Container Toolkit: https://github.com/NVIDIA/nvidia-docker. This allows you to use NVIDIA GPUs in a docker container.

- NVIDIA Driver >= 450 (currently, may change in the future) installed on the host machine.
You can check which version is currently installed by running: `nvidia-smi | grep "Driver Version:"`

- ImageNet Dataset in TFRecords format: http://www.image-net.org/
To download and preprocess the dataset, use the [Generate ImageNet for TensorFlow script](https://github.com/tensorflow/models/blob/master/research/inception/inception/data/download_and_preprocess_imagenet.sh). The dataset will be downloaded to a directory specified as the first parameter of the script.

For the rest of this blog post, we will assume all of the later requirements are satisfied.

## 1. Training Resnet50v1.5 using the TF1.x container

First things first we want to download the latest release of the Resnet50v1.5 code on our machine:

```bash
# Create a directory, so Git doesn't get messy, and enter it
$ mkdir resnet50v1.5 && cd resnet50v1.5

# Initialize a git repository
$ git init

# Adding the git remote and fetch the existing branches
$ git remote add -f origin  https://github.com/NVIDIA/DeepLearningExamples.git

# Enable the tree check feature
$ git config core.sparseCheckout true

# We specify that we want to only pull a subdirectory from origin
$ echo 'TensorFlow/Classification/ConvNets/' >> .git/info/sparse-checkout

## Execute the `git pull` against the remote repository
$ git pull origin master

## Move all files to our working directory and clean leftovers
$ mv TensorFlow/Classification/ConvNets/* . && rm -rf TensorFlow .git

# List the files and directories present in our working directory
$ ls -al

rw-rw-r--  user  user   203 B    Tue Aug 18 20:09:41 2020    Dockerfile
rw-rw-r--  user  user     4 KiB  Tue Aug 18 20:09:41 2020    export_frozen_graph.py
rw-rw-r--  user  user    11 KiB  Tue Aug 18 20:09:41 2020    LICENSE
rwxrwxr-x  user  user     4 KiB  Tue Aug 18 20:09:41 2020    main.py
rwxrwxr-x  user  user     4 KiB  Tue Aug 18 20:09:41 2020    model/
rw-rw-r--  user  user     3 KiB  Tue Aug 18 20:09:41 2020    postprocess_ckpt.py
rw-rw-r--  user  user     3 KiB  Tue Aug 18 20:09:41 2020    README.md
rw-rw-r--  user  user    48 B    Tue Aug 18 20:09:41 2020    requirements.txt
rwxrwxr-x  user  user     4 KiB  Tue Aug 18 20:09:41 2020    resnet50v1.5/
rwxrwxr-x  user  user     4 KiB  Tue Aug 18 20:09:41 2020    resnext101-32x4d/
rwxrwxr-x  user  user     4 KiB  Tue Aug 18 20:09:41 2020    runtime/
rwxrwxr-x  user  user     4 KiB  Tue Aug 18 20:09:41 2020    se-resnext101-32x4d/
rwxrwxr-x  user  user     4 KiB  Tue Aug 18 20:09:41 2020    utils/
```

Now that we have all the files we need in our environments. We will start the TF1.x container to start training:

```bash
$ docker run -it --rm \
   --gpus="all" \
   --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
   --workdir /workspace/ \
   -v "$(pwd):/workspace/" \
   -v "</path/to/imagenet/tfrecords/>:/data/" \   # Update the path with the correct one here
   nvcr.io/nvidia/tensorflow:20.07-tf1-py3
```

<!--
docker run -it --rm --network=host \
   --gpus="device=1" \
   --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
   --workdir /workspace/ \
   -v "$(pwd):/workspace/" \
   -v "${RAID_STORAGE_PATH}/datasets/imagenet/tfrecords:/data/" \
   nvcr.io/nvidia/tensorflow:20.07-tf1-py3
-->

Let's verify we can access all the files we need:

```bash
# Let's first test that we can access the Resnet50v1.5 code that we previously downloaded
$ ls -al
-rw-rw-r-- 1 1000 1000   203 Aug 19 03:09 Dockerfile
-rw-rw-r-- 1 1000 1000 11358 Aug 19 03:09 LICENSE
-rw-rw-r-- 1 1000 1000  3356 Aug 19 03:09 README.md
-rw-rw-r-- 1 1000 1000  4662 Aug 19 03:09 export_frozen_graph.py
-rwxrwxr-x 1 1000 1000  5047 Aug 19 03:09 main.py
drwxrwxr-x 4 1000 1000  4096 Aug 19 03:09 model
-rw-rw-r-- 1 1000 1000  3326 Aug 19 03:09 postprocess_ckpt.py
-rw-rw-r-- 1 1000 1000    48 Aug 19 03:09 requirements.txt
drwxrwxr-x 4 1000 1000  4096 Aug 19 03:09 resnet50v1.5
drwxrwxr-x 4 1000 1000  4096 Aug 19 03:09 resnext101-32x4d
drwxrwxr-x 2 1000 1000  4096 Aug 19 03:09 runtime
drwxrwxr-x 4 1000 1000  4096 Aug 19 03:09 se-resnext101-32x4d
drwxrwxr-x 3 1000 1000  4096 Aug 19 03:09 utils


# Let's verify that we access the training data
$ ls -1 /data/train* | sort | head -5
>>> /data/train-00000-of-01024
>>> /data/train-00001-of-01024
>>> /data/train-00002-of-01024
>>> /data/train-00003-of-01024
>>> /data/train-00004-of-01024

# Let's verify that we access the inference data
$ ls -1 /data/val* | sort | head -5
>>> /data/validation-00000-of-00128
>>> /data/validation-00001-of-00128
>>> /data/validation-00002-of-00128
>>> /data/validation-00003-of-00128
>>> /data/validation-00004-of-00128

# Let's verify we can see our GPUs:
$ nvidia-smi

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.XX.XX    Driver Version: 450.XX.XX    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Quadro GV100        Off  | 00000000:1E:00.0 Off |                  Off |
| 38%   52C    P2    30W / 250W |      1MiB / 32508MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

Now we want to train Resnet50v1.5. Warning this process can take some time.
To accelerate training time we will use [AMP (automatic mixed precision)](https://developer.nvidia.com/automatic-mixed-precision) and [XLA (Tensorflow JIT Compiler)](https://www.tensorflow.org/xla).

To be noted: This script will automatically use all the GPUs you have available in your machine.

```bash
# Install the dependencies we need
pip install -r requirements.txt

# Collect the number of NVIDIA GPUs available in the machine
N_GPUS=$(nvidia-smi -L 2>/dev/null| wc -l || echo 1)

# Set the MPI command appropriately if there is more than one GPU in the machine
if [[ "$N_GPUS" -gt "1" ]]; then
   MPI_COMMAND="mpiexec --allow-run-as-root --bind-to socket -np ${N_GPUS}"
else
   MPI_COMMAND=""
fi

# Launch Training for Resnet50v1.5
${MPI_COMMAND} python3 main.py \
    --arch=resnet50 \
    --mode=train_and_evaluate \
    --iter_unit=epoch \
    --num_iter=90 \
    --batch_size=256 \
    --warmup_steps=100 \
    --use_cosine \
    --label_smoothing 0.1 \
    --lr_init=0.256 \
    --lr_warmup_epochs=8 \
    --momentum=0.875 \
    --weight_decay=3.0517578125e-05 \
    --use_tf_amp \
    --use_static_loss_scaling \
    --loss_scale 128 \
    --data_dir=/data/ \
    --results_dir=./results \
    --weight_init=fan_in
```

Once the model has finished training we can now starts the next phase. In order to be usable we need to convert the trained model saved as a [TensorFlow checkpoint](https://www.tensorflow.org/guide/checkpoint) to a format called [TensorFlow SavedModel](https://www.tensorflow.org/guide/saved_model) which is readable by TF-TRT.

For this let's create a file called `export_to_saved_model.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import horovod.tensorflow as hvd  # Necessary to register Horovod OPs that may exists

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.saved_model import tag_constants
from tensorflow.tools.graph_transforms import TransformGraph

PATH_TO_CHECKPOINT = "./results"


def optimize_graph_for_inference(model_dir, input_node_names, output_node_names):
    input_graph_path = os.path.join(model_dir, 'graph.pbtxt')
    input_checkpoint = tf.train.latest_checkpoint(model_dir)

    input_binary = False
    clear_devices = True

    print("Loading Graph `{}` ...".format(input_graph_path))
    print("Loading Checkpoint `{}` ...".format(input_checkpoint))

    print("Freezing Graph ...")
    # https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/python/tools/freeze_graph.py#L286-L301
    frozen_graph_def = freeze_graph.freeze_graph(
        input_graph=input_graph_path,  # A `GraphDef` file to load.
        input_saver="",  #  A TensorFlow Saver file.
        input_binary=input_binary,  #  A Bool. True means input_graph is .pb, False indicates .pbtxt.
        input_checkpoint=input_checkpoint,
        output_node_names=",".join(output_node_names),  # The name(s) of the output nodes, comma separated.
        restore_op_name="",  # Unused
        filename_tensor_name="",  # Unused
        output_graph=os.path.join("/tmp", 'frozen_saved_model.pb'),  # String where to write the frozen `GraphDef`.
        clear_devices=clear_devices,  #  A Bool whether to remove device specifications.
        initializer_nodes="",  # Comma separated list of initializer nodes to run before freezing.
        variable_names_whitelist="",  # The set of variable names to convert (optional, bydefault, all variables are converted)
        variable_names_blacklist="",  # The set of variable names to omit converting to constants (optional)
        input_meta_graph=None,  # A `MetaGraphDef` file to load (optional).
        input_saved_model_dir=None,  # Path to the dir with TensorFlow 'SavedModel' file and variables (optional).
        saved_model_tags=tag_constants.SERVING,  # Group of comma separated tag(s) of the MetaGraphDef to load, in string format.
        checkpoint_version=saver_pb2.SaverDef.V2  # Tensorflow variable file format (saver_pb2.SaverDef.V1 or saver_pb2.SaverDef.V2
    )

    print("Optimizing Graph for Inference ...")
    optimized_frozen_graph = optimize_for_inference_lib.optimize_for_inference(
            frozen_graph_def,
            input_node_names,  # an array of the input node(s)
            output_node_names,  # an array of output nodes
            tf.float32.as_datatype_enum
    )

    transforms = [
        'remove_nodes(op=Identity)',
        'merge_duplicate_nodes',
        'strip_unused_nodes',
        'fold_constants(ignore_errors=true)',
        'fold_batch_norms'
    ]

    print("Applying Graph Transformations ...")
    return TransformGraph(
        optimized_frozen_graph,
        input_node_names,  # an array of the input node(s)
        output_node_names,  # an array of output nodes
        transforms
    )


def convert_graph_def_to_saved_model(
    export_dir,
    graphdef_file,
    input_nodes=None,
    output_nodes=None
):
  if tf.io.gfile.exists(export_dir):
    tf.io.gfile.rmtree(export_dir)

  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graphdef_file, name='')

    print('Exporting Optimized SavedModel to Disk: `{}` ...'.format(export_dir))
    tf.saved_model.simple_save(
        sess,
        export_dir,
        inputs={
            name: sess.graph.get_tensor_by_name('{}:0'.format(node_name))
            for name, node_name in input_nodes
        },
        outputs={
            name: sess.graph.get_tensor_by_name('{}:0'.format(node_name))
            for name, node_name in output_nodes
        }
    )
    print('Optimized graph converted to SavedModel!')


if __name__ == "__main__":
    tftrt_ready_savedmodel_dir = os.path.join(PATH_TO_CHECKPOINT, "tftrt_ready")

    try:
        os.makedirs(tftrt_ready_savedmodel_dir)
    except FileExistsError:
        pass

    input_nodes = [("input", "StagingArea_1_get")]
    output_nodes = [
        ("probs", "resnet50/output/softmax"),
        ("logits", "resnet50/output/dense/BiasAdd")
    ]

    inference_graph_def = optimize_graph_for_inference(
        model_dir=PATH_TO_CHECKPOINT,
        input_node_names=[node_name for _, node_name in input_nodes],
        output_node_names=[node_name for _, node_name in output_nodes]
    )

    convert_graph_def_to_saved_model(
        tftrt_ready_savedmodel_dir,
        graphdef_file=inference_graph_def,
        input_nodes=input_nodes,
        output_nodes=output_nodes
    )
```

Let's now execute the script:

```bash
$ python export_to_saved_model.py

>>> Loading Graph `./results/graph.pbtxt` ...
>>> Loading Checkpoint `./results/nvidia_rn50_tf_amp` ...
>>> Freezing Graph ...
>>> Optimizing Graph for Inference ...
>>> Applying Graph Transformations ...
>>> Exporting Optimized SavedModel to Disk: `./results/tftrt_ready` ...
>>> Optimized graph converted to SavedModel!
```

Let's now verify that our SavedModel is readable and correct:
```bash
$ saved_model_cli show --dir results/tftrt_ready/ --all

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input'] tensor_info:
        dtype: DT_FLOAT
        shape: (256, 224, 224, 3)
        name: StagingArea_1_get:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['logits'] tensor_info:
        dtype: DT_FLOAT
        shape: (256, 1001)
        name: resnet50/output/dense/BiasAdd:0
    outputs['probs'] tensor_info:
        dtype: DT_FLOAT
        shape: (256, 1001)
        name: resnet50/output/softmax:0
  Method name is: tensorflow/serving/predict
```

Now that we have our model saved in the correct format: [TensorFlow SavedModel](https://www.tensorflow.org/guide/saved_model). We can now switch to Tensorflow 2.x container and proceed with the inference phase.


## 2. Inferencing Resnet50v1.5 using the TF2.x container

Let's quit the previous container with command `exit` and start the TensorFlow 2.x container:

```bash
docker run -it --rm \
   --gpus="all" \
   --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
   --workdir /workspace/ \
   -v "$(pwd):/workspace/" \
   -v "</path/to/imagenet/tfrecords/>:/data/" \   # Update the path with the correct one here
   nvcr.io/nvidia/tensorflow:20.07-tf2-py3
```

<!--
docker run -it --rm --network=host \
   --gpus="device=1" \
   --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
   --workdir /workspace/ \
   -v "$(pwd):/workspace/" \
   -v "${RAID_STORAGE_PATH}/datasets/imagenet/tfrecords:/data/" \
   nvcr.io/nvidia/tensorflow:20.07-tf2-py3
-->

Again, let's verify we have access to all the files we need:

Let's verify we can access all the files we need:

```bash
# Let's first test that we can access the Resnet50v1.5 code that we previously downloaded
ls -al
-rw-rw-r--  1 1000 1000   203 Aug 19 03:09 Dockerfile
-rw-rw-r--  1 1000 1000 11358 Aug 19 03:09 LICENSE
-rw-rw-r--  1 1000 1000  3356 Aug 19 03:09 README.md
-rw-rw-r--  1 1000 1000  4688 Aug 19 20:36 export_frozen_graph.py
-rw-rw-r--  1 1000 1000  3653 Aug 19 21:30 export_to_saved_model.py
-rwxrwxr-x  1 1000 1000  5047 Aug 19 03:09 main.py*
drwxrwxr-x  5 1000 1000  4096 Aug 19 03:13 model/
-rw-rw-r--  1 1000 1000  3326 Aug 19 03:09 postprocess_ckpt.py
-rw-rw-r--  1 1000 1000    48 Aug 19 03:09 requirements.txt
drwxrwxr-x  4 1000 1000  4096 Aug 19 03:09 resnet50v1.5/
drwxrwxr-x  4 1000 1000  4096 Aug 19 03:09 resnext101-32x4d/
drwxrwxr-x  3 1000 1000  4096 Aug 19 21:30 results/
drwxrwxr-x  3 1000 1000  4096 Aug 19 03:13 runtime/
drwxrwxr-x  4 1000 1000  4096 Aug 19 03:09 se-resnext101-32x4d/
drwxrwxr-x  4 1000 1000  4096 Aug 19 03:13 utils/



# Let's verify that we access the training data
ls -1 /data/train* | sort | head -5
>>> /data/train-00000-of-01024
>>> /data/train-00001-of-01024
>>> /data/train-00002-of-01024
>>> /data/train-00003-of-01024
>>> /data/train-00004-of-01024

# Let's verify that we access the inference data
ls -1 /data/val* | sort | head -5
>>> /data/validation-00000-of-00128
>>> /data/validation-00001-of-00128
>>> /data/validation-00002-of-00128
>>> /data/validation-00003-of-00128
>>> /data/validation-00004-of-00128

# Let's verify we can see our GPUs:
nvidia-smi

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.XX.XX    Driver Version: 450.XX.XX    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Quadro GV100        Off  | 00000000:1E:00.0 Off |                  Off |
| 38%   52C    P2    30W / 250W |      1MiB / 32508MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

We should also be able to visualize properly the saved model we generated at the
previous step:
```bash
$ saved_model_cli show --dir results/tftrt_ready/ --all

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input'] tensor_info:
        dtype: DT_FLOAT
        shape: (256, 224, 224, 3)
        name: StagingArea_1_get:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['logits'] tensor_info:
        dtype: DT_FLOAT
        shape: (256, 1001)
        name: resnet50/output/dense/BiasAdd:0
    outputs['probs'] tensor_info:
        dtype: DT_FLOAT
        shape: (256, 1001)
        name: resnet50/output/softmax:0
  Method name is: tensorflow/serving/predict
```

Now that we ensured that everything is the way we expected to be, let's install the dependencies we will need:

```bash
pip install -r requirements.txt 
```

Then, let's import the model we trained with TensorFlow 1.x into TensorFlow 2.x and execute inference. For this we will use the following python file called `tf2_inference.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import copy
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

SAVEDMODEL_PATH = "./results/tftrt_ready"

def load_and_convert(path, precision):
    """ Load a saved model and convert it to FP32 or FP16. Return a converter """

    params = copy.deepcopy(trt.DEFAULT_TRT_CONVERSION_PARAMS)

    params = params._replace(
        precision_mode=(
            trt.TrtPrecisionMode.FP16
            if precision.lower() == "fp16" else
            trt.TrtPrecisionMode.FP32
        ),
        max_batch_size=128,
        max_workspace_size_bytes=2 << 32,  # 8,589,934,592 bytes
        maximum_cached_engines=100,
        minimum_segment_size=3,
        is_dynamic_op=True,
        allow_build_at_runtime=True
    )

    import pprint
    print("%" * 85)
    pprint.pprint(params)
    print("%" * 85)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=path,
        conversion_params=params,
    )

    return converter


if __name__ == "__main__":
    converter = load_and_convert(
        os.path.join(SAVEDMODEL_PATH),
        precision="fp16"
    )
    xx = converter.convert()
    converter.save(
        os.path.join(SAVEDMODEL_PATH, "converted")
    )

    def dataloader_fn(filenames, batch_size, height, width):

        ds = tf.data.Dataset.from_tensor_slices(filenames)

        ds = ds.apply(
            tf.data.experimental.parallel_interleave(
                tf.data.TFRecordDataset,
                cycle_length=10,
                block_length=8,
                prefetch_input_elements=16
            )
        )

        counter = tf.data.Dataset.range(sys.maxsize)
        ds = tf.data.Dataset.zip((ds, counter))

        import imp
        image_processing = imp.load_source('', './utils/image_processing.py')

        def preprocess_image_record(record, height, width):

            def _deserialize_image_record(record):
                feature_map = {
                    'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
                    'image/class/label': tf.io.FixedLenFeature([], tf.int64, -1),
                    'image/class/text': tf.io.FixedLenFeature([], tf.string, ''),
                    'image/object/bbox/xmin': tf.io.VarLenFeature(
                        dtype=tf.float32),
                    'image/object/bbox/ymin': tf.io.VarLenFeature(
                        dtype=tf.float32),
                    'image/object/bbox/xmax': tf.io.VarLenFeature(
                        dtype=tf.float32),
                    'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32)
                }
                with tf.name_scope('deserialize_image_record'):
                    obj = tf.io.parse_single_example(record, feature_map)
                    imgdata = obj['image/encoded']
                    label = tf.cast(obj['image/class/label'], tf.int32)
                    bbox = tf.stack(
                        [obj['image/object/bbox/%s' % x].values for x in
                         ['ymin', 'xmin', 'ymax', 'xmax']])
                    bbox = tf.transpose(tf.expand_dims(bbox, 0), [0, 2, 1])
                    text = obj['image/class/text']
                    return imgdata, label, bbox, text

            imgdata, label, bbox, text = _deserialize_image_record(record)
            label -= 1

            try:
                image = image_processing._decode_jpeg(imgdata, channels=3)
            except:
                image = tf.image.decode_image(imgdata, channels=3)

            def _aspect_preserving_resize(image, resize_min):
                """Resize images preserving the original aspect ratio.

                Args:
                image: A 3-D image `Tensor`.
                resize_min: A python integer or scalar `Tensor` indicating the size of
                  the smallest side after resize.

                Returns:
                resized_image: A 3-D tensor containing the resized image.
                """
                shape = tf.shape(image)
                height, width = shape[0], shape[1]

                new_height, new_width = image_processing._smallest_size_at_least(
                    height, width, resize_min
                )

                return tf.image.resize(
                    image,
                    [new_height, new_width],
                    method=tf.image.ResizeMethod.BILINEAR
                )

            image = _aspect_preserving_resize(image, 256)
            image = image_processing._central_crop(image, height, width)

            return image, label

        ds = ds.apply(
            tf.data.experimental.map_and_batch(
                map_func=lambda record, _: preprocess_image_record(
                    record, height, width
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                batch_size=batch_size,
                drop_remainder=True,
            )
        )

        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds

    filename_pattern = os.path.join("/data", 'validation-*')
    filenames = sorted(glob.glob(filename_pattern))

    BATCH_SIZE = 128

    ds = dataloader_fn(
        filenames=filenames,
        batch_size=BATCH_SIZE,
        height=224,
        width=224
    )
    iterator = iter(ds)
    features, labels = iterator.get_next()

    root = tf.saved_model.load(os.path.join(SAVEDMODEL_PATH, "converted"))
    concrete_func = root.signatures['serving_default']

    from statistics import mean
    import time

    try:
        step_times = list()
        for step in range(200):
            print("Processing step: %03d ..." % (step + 1))
            start_t = time.time()
            probs = concrete_func(features)
            if step >= 50:
                step_times.append(time.time() - start_t)
    except tf.errors.OutOfRangeError:
        pass

    avg_step_time = mean(step_times)
    print("Average step time: %.1f msec" % (avg_step_time * 1e3))
    print("Average throughput: %d samples/sec" % (
        BATCH_SIZE / avg_step_time
    ))
```

Now when we execute:
```bash
$ python tf2_inference.py

Processing step: 001 ...
Processing step: 002 ...
Processing step: 003 ...
Processing step: 004 ...
Processing step: 005 ...
[...]
Processing step: 195 ...
Processing step: 196 ...
Processing step: 197 ...
Processing step: 198 ...
Processing step: 199 ...
Processing step: 200 ...
Average step time: 146.4 msec
Average throughput: 874 samples/sec:
```

This run has been executed with an NVIDIA GPU QUADRO GV100-32GB, your throughput results may vary.

With this article we have shown that you can use your model trained in TF1.x and do inference with the latest performance optimizations that come with TF2.x
releases. And all of this can be done with a minimal number of changes and does not impact in any form the training pipeline.