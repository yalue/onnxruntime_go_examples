Getting ONNX Input and Output Information
=========================================

This example project defines a command-line utility that prints the input and
output information about a user-specified .onnx file to standard output.

Example usage:
```
go build .

./onnx_list_inputs_and_outputs -onnx_file ../image_object_detect/yolov8n.onnx
```

The above command should output something like the following:

```
1 inputs to ../image_object_detect/yolov8n.onnx:
  Index 0: "images": [1 3 640 640], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
1 outputs from ../image_object_detect/yolov8n.onnx:
  Index 0: "output0": [1 84 8400], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
```

(The yolov8 network only has one input and one output: a 1x3x640x640 input,
named "images", and a 1x84x8400 output, named "output0".)

