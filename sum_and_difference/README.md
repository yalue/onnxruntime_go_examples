Sum and Difference `onnxruntime_go` Example
===========================================

This is a basic, heavily-commented command-line program that uses the
`onnxruntime_go` library to load and run an ONNX-format neural network.

Usage
-----

Build the program using `go build`. After this, it should run without arguments
on most systems: `./sum_and_difference`.  If you encounter errors, you may need
to specify a different version of the `onnxruntime` shared library, using the
`-onnxruntime_lib` command-line flag.  (Run the program with `-help` to see
usage information.)

```bash
go build .
./sum_and_difference
```

Should output the following if successful:
```
The network ran without errors.
  Input data: [0.2 0.3 0.6 0.9]
  Approximate sum of inputs: 1.999988
  Approximate max difference between any two inputs: 0.607343
The network seemed to run OK!
```

