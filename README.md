Example Applications for the `onnxruntime_go` Library
=====================================================

This repository contains a collection of (mostly) simple standalone examples
using the [`onnxruntime_go`](https://github.com/yalue/onnxruntime_go) library
to run neural-network applications.


Prerequisites
-------------

You will need to be using a version of Go with cgo enabled---meaning that on
Windows you'll need to have `gcc` available on your PATH.

If you wish to use hardware acceleration such as CUDA, you'll need to have a
compatible version of the `onnxruntime` library compiled with support for your
platform of choice. CoreML should almost always be available on Apple hardware,
but other supported acceleration frameworks (e.g., TensorRT or CUDA) may have
additional prerequisites, which are documented in
[the official onnxruntime documentation](https://onnxruntime.ai/docs/execution-providers/).
Note that not all execution providers supported by `onnxruntime` itself are
supported by `onnxruntime_go`.

The `onnxruntime` shared libraries for some common platforms are included
under the `third_party/` directory in this repository.


Usage
-----

Navigate to any one of the subdirectories, and run `go build` to produce an
executable on your system.  Many executables will provide a mechanism for
specifying a path to an `onnxruntime` shared library file.  For example:

```bash
cd sum_and_difference
go build

# You can specify any version of the onnxruntime library here, but this would
# be the correct library version on 64-bit AMD or Intel Linux systems.
./sum_and_difference --onnxruntime_lib ../third_party/onnxruntime.so
```

Be aware that different examples may use different mechanisms for locating the
correct shared library version.


List of Examples
----------------

 - `sum_and_difference`: This is the simplest example, copied from a unit test
   in the `onnxruntime_go` library.  It uses a basic neural network (trained
   using a pytorch script contained in the directory) on a tiny amount of
   hardcoded data.  The source code is very heavily commented for reference.

 - `mnist`: This example runs a CNN trained to identify handwritten digits from
   the MNIST dataset. It processes a single input image, and outputs the digit
   it is most likely to contain.

 - `image_object_detect`: This example uses the YOLOv8 network to detect a list
   of objects in an input image. It also attempts to use CoreML if the
   `USE_COREML` environment variable is set to `"true"`.

