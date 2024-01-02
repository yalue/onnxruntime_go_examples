`onnxruntime_go`: Float16 MNIST Example
=======================================

This example is nearly identical to the plain `mnist` example from this
repository, but uses a model that has been converted to use 16-bit floats. This
example is intended to illustrate how to convert inputs to 16-bit floating
point values using the `github.com/x448/float16` package and the
`CustomDataTensor` type from `onnxruntime_go`.

The code has been mostly copied and pasted from the `../mnist` example. It
differs only in a few places:

 - The `ProcessedImage.GetNetworkInput` function now converts each input pixel
   from a float32 grayscale value to a float16, and writes the float16 data
   into a slice of bytes.

 - The `input` and `output` tensors created in the `classifyDigit` function are
   now `CustomDataTensor`s, backed by slices of bytes.

 - The `convertFloat16Data` function has been added to convert the output
   tensor's bytes from `float16.Float16` data to a slice of `float32`s.

The included `mnist_float16.onnx` network was created by using the
`onnxconverter-common` python package on the `../mnist/mnist.onnx` network,
using the process described on
[this page](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html).

Example Usage
-------------

This program is used in the exact same way as `../mnist`. Build it using
`go build`, and run it with `-help` to see all command-line flags. It loads
`mnist_float16.onnx` from the current directory.

For example,
```bash
go build .
./mnist_float16 ../mnist/eight.png
```

Will produce the following output:
```
Saved postprocessed input image to ./postprocessed_input_image.png.
  0: 1.350586
  1: 1.148438
  2: 2.232422
  3: 0.827148
  4: -3.474609
  5: 1.199219
  6: -1.187500
  7: -5.960938
  8: 4.765625
  9: -2.345703
../mnist/eight.png is probably a 8, with probability 4.765625
Everything seemed to run OK!
```

