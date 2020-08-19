# How to train with Tensorflow 1.x and perform inference with Tensorflow 2.x using TF-TRT ?

**NOTE: WE SHALL PROBABLY EXPLAIN WHY WE WANT PEOPLE TO USE TF2.x. Long story short: TF1.x TF-TRT is not updated any more, a number of optimizations and new features are not available in TF1.x (e.g. Dynamic Shape, new layers are supported, etc.)**.

This blogpost will focus on explaining how to to use a model trained with TF1.x and execute inference in TF2.x. In this sense, we will use [Resnet50v1.5](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5) published as part of the [NVIDIA Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples) github repository. This process can be extrapolated to any other models.

In this blogpost we will use the latest available tensorflow containers (at the time of writing):

* **Tensorflow 1.x Container**: `docker pull nvcr.io/nvidia/tensorflow:20.07-tf1-py3`
* **Tensorflow 2.x Container**: `docker pull nvcr.io/nvidia/tensorflow:20.07-tf2-py3`

Prerequesites to be able to follow this blog posts/tutorial:
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
mkdir resnet50v1.5 && cd resnet50v1.5

# Initialize a git repository
git init

# Adding the git remote and fetch the existing branches
git remote add -f origin  https://github.com/NVIDIA/DeepLearningExamples.git

# Enable the tree check feature
git config core.sparseCheckout true

# We specify that we want to only pull a subdirectory from origin
echo 'TensorFlow/Classification/ConvNets/' >> .git/info/sparse-checkout

## Execute the `git pull` against the remote repository
git pull origin master

## Move all files to our working directory and clean leftovers
mv TensorFlow/Classification/ConvNets/* . && rm -rf TensorFlow .git

# List the files and directories present in our working directory
ls -al

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
docker run -it --rm --network=host \
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
ls -al
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

Once the model has finished training we can now starts the next phase: inference using TF2.x