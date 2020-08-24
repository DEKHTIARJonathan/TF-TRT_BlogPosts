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

