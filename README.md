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

 - `mnist_float16`: This example is identical to the plain `mnist` example,
   except it uses a 16-bit network, including 16-bit inputs and outputs. It is
   intended to illustrate how to use a float16 `CustomDataTensor`.

 - `onnx_list_inputs_and_outputs`: This example prints the inputs and outputs
   of a user-specified .onnx file to stdout. It is intended to illustrate the
   usage of the `onnxruntime_go.GetInputOutputInfo` function.

 - `image_object_detect`: This example uses the YOLOv8 network to detect a list
   of objects in an input image. It also attempts to use CoreML if the
   `USE_COREML` environment variable is set to `true`.

 - `non_tensor_outputs`: This example runs a network produced by the `sklearn`
   python library, which is notable for outputting ONNX `Map` and `Sequence`
   types. This example is meant to serve as a reference for how users may
   access `Map` and `Sequence` contents.

Contributing and Opening New Issues
-----------------------------------

PRs with new examples to this repository are welcome.  Each example should be
in its own subdirectory with its own go.mod file, and include only minimal
dependencies (i.e., do not include several hundred megabytes of .onnx files or
data). Each example should include a README, be formatted using `gofmt`, and
contain ample comments to serve as a useful example to other users.

Please limit open issues in this repository to bugs with existing examples.
Issues are _not_ a place to request help with `onnxruntime` in general. Such
issues will be ignored going forward.  If you have not run your `.onnx` network
using the `onnxruntime` library in python, this is not the place to get help
with it.  Learning to use `onnxruntime` in python is easier than in Go, and
will give a point of reference that you understand the network you are trying to
run, and that your inputs and outputs are correct.

In short, this repository is intended to provide examples for using the
`onnxruntime_go` wrapper in specific.  Users are expected to already understand
`.onnx` files and how to use `onnxruntime` in general.
