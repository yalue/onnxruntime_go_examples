`onnxruntime_go`: MNIST Example
===============================

This example makes use of the pre-trained MNIST network, obtained from the
[official ONNX models repository](https://github.com/onnx/models/tree/ddbbd1274c8387e3745778705810c340dea3d8c7/validated/vision/classification/mnist).
Specifically, the included `mnist.onnx` is MNIST-12 from the above link.

This example uses the network to analyze single image files specified on the
command line.

Example Usage
-------------

Run the program with `-help` to see all command-line flags. In general, you
will need to supply it with an input image.

```bash
./mnist -image_path ./eight.png
./mnist -image_path ./tiny_5.png

# There's an additional flag if you want to invert the image colors. The
# network is trained on images with black backgrounds, so you may want to
# invert images with white backgrounds.
./mnist -image_path ./seven.png -invert_image
```

Note that the program will also create `postprocessed_input_image.png` in the
current directory, showing the image that was passed to the neural network
after resizing and converting to grayscale.

