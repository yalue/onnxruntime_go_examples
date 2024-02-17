Image Object Detection Using Yolo
=================================

This example uses the included yolov8n.onnx network to detect images in an
image. For now, the example is hardcoded to process the included car.png image.
It performs the detection several times in order to compute timing statistics.


CoreML can be enabled by setting the `USE_COREML` environment variable to
`true`. (Though this will cause the program to fail on systems where CoreML is
not supported.)

Running with CoreML
-------------------
```bash
$ USE_COREML=true ./run.sh

Object: car Confidence: 0.50 Coordinates: (392.156250, 286.328125), (692.111755, 655.371094)
Object: car Confidence: 0.50 Coordinates: (392.156250, 286.328125), (692.111755, 655.371094)
Object: car Confidence: 0.50 Coordinates: (392.156250, 286.328125), (692.111755, 655.371094)
Object: car Confidence: 0.50 Coordinates: (392.156250, 286.328125), (692.111755, 655.371094)
Object: car Confidence: 0.50 Coordinates: (392.156250, 286.328125), (692.111755, 655.371094)
Min Time: 17.401875ms, Max Time: 21.7065ms, Avg Time: 19.258691ms, Count: 5
50th: 18.485666ms, 90th: 21.7065ms, 99th: 21.7065ms
```

Run on the CPU only, without CoreML
-----------------------------------
```bash
$ ./run.sh

Object: car Confidence: 0.50 Coordinates: (392.655396, 285.742920), (691.901306, 656.455566)
Object: car Confidence: 0.50 Coordinates: (392.655396, 285.742920), (691.901306, 656.455566)
Object: car Confidence: 0.50 Coordinates: (392.655396, 285.742920), (691.901306, 656.455566)
Object: car Confidence: 0.50 Coordinates: (392.655396, 285.742920), (691.901306, 656.455566)
Object: car Confidence: 0.50 Coordinates: (392.655396, 285.742920), (691.901306, 656.455566)
Min Time: 41.5205ms, Max Time: 58.348084ms, Avg Time: 46.154341ms, Count: 5
50th: 43.471958ms, 90th: 58.348084ms, 99th: 58.348084ms
```
(Note the slower execution times.)
