## tensorflow2_cpp

Sample example how to load a Tensorflow Object detection API v2 model and serve prediction in C++  

## Build and installation

The current config uses the following dependencies (based on Tensorflow tested build). Check out [build from source configs](https://www.tensorflow.org/install/source#gpu) for more details.


1. Tensorflow 2.3.0
2. CUDA 10.1
3. cuDNN 7.6
4. Bazel 3.1.0
5. Protobuf 3.9.2
6. OpenCV 4.3.0 (required only for the example)

### Build

```bash
docker build . -t boraraktim/tensorflow2_cpp
```

OR

`docker pull boraraktim/tensorflow2_cpp`

### Compile

```bash
# Start docker container
docker run --gpus all -it --rm -v efficientdet_d3_coco17_tpu-32/:/object_detection/models/ boraraktim/tensorflow2_cpp
make build_cpp
```

directory structure

```
-|efficientdet_d3_coco17_tpu-32
    |--saved_model
        |--assets/
        |--saved_model.pb
        |-- ...

```

### Predict

```
./get_prediction <path/to/saved_model> <path/to/image.jpg> <path/to/output.jpg>
```

![sample_prediction_doggies.jpg](./sample_prediction.jpg)

Image from [Unspalsh](https://unsplash.com/photos/2_3c4dIFYFU)
