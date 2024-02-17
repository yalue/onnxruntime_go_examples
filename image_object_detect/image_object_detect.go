package main

import (
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"runtime"
	"sort"

	"github.com/8ff/prettyTimer"
	"github.com/nfnt/resize"
	ort "github.com/yalue/onnxruntime_go"
)

var modelPath = "./yolov8n.onnx"
var imagePath = "./car.png"
var useCoreML = false

type ModelSession struct {
	Session *ort.AdvancedSession
	Input   *ort.Tensor[float32]
	Output  *ort.Tensor[float32]
}

func main() {
	os.Exit(run())
}

func run() int {
	timingStats := prettyTimer.NewTimingStats()

	if os.Getenv("USE_COREML") == "true" {
		useCoreML = true
	}

	// Read the input image into a image.Image object
	pic, e := loadImageFile(imagePath)
	if e != nil {
		fmt.Printf("Error loading input image: %s\n", e)
		return 1
	}
	originalWidth := pic.Bounds().Canon().Dx()
	originalHeight := pic.Bounds().Canon().Dy()

	modelSession, e := initSession()
	if e != nil {
		fmt.Printf("Error creating session and tensors: %s\n", e)
		return 1
	}
	defer modelSession.Destroy()

	// Run the detection 5 times
	for i := 0; i < 5; i++ {
		e = prepareInput(pic, modelSession.Input)
		if e != nil {
			fmt.Printf("Error converting image to network input: %s\n", e)
			return 1
		}

		timingStats.Start()
		e = modelSession.Session.Run()
		if e != nil {
			fmt.Printf("Error running ORT session: %s\n", e)
			return 1
		}
		timingStats.Finish()

		// Print the results
		boxes := processOutput(modelSession.Output.GetData(), originalWidth,
			originalHeight)
		for i, box := range boxes {
			fmt.Printf("Box %d: %s\n", i, &box)
		}
	}
	timingStats.PrintStats()
	return 0
}

func loadImageFile(filePath string) (image.Image, error) {
	f, e := os.Open(filePath)
	if e != nil {
		return nil, fmt.Errorf("Error opening %s: %w", filePath, e)
	}
	defer f.Close()
	pic, _, e := image.Decode(f)
	if e != nil {
		return nil, fmt.Errorf("Error decoding %s: %w", filePath, e)
	}
	return pic, nil
}

// Populates a yolov8n input tensor with the contents of the given image.
func prepareInput(pic image.Image, dst *ort.Tensor[float32]) error {
	data := dst.GetData()
	channelSize := 640 * 640
	if len(data) < (channelSize * 3) {
		return fmt.Errorf("Destination tensor only holds %d floats, needs "+
			"%d (make sure it's the right shape!)", len(data), channelSize*3)
	}
	redChannel := data[0:channelSize]
	greenChannel := data[channelSize : channelSize*2]
	blueChannel := data[channelSize*2 : channelSize*3]

	// Resize the image to 640x640 using Lanczos3 algorithm
	pic = resize.Resize(640, 640, pic, resize.Lanczos3)
	i := 0
	for y := 0; y < 640; y++ {
		for x := 0; x < 640; x++ {
			r, g, b, _ := pic.At(x, y).RGBA()
			redChannel[i] = float32(r>>8) / 255.0
			greenChannel[i] = float32(g>>8) / 255.0
			blueChannel[i] = float32(b>>8) / 255.0
			i++
		}
	}

	return nil
}

func getSharedLibPath() string {
	if runtime.GOOS == "windows" {
		if runtime.GOARCH == "amd64" {
			return "../third_party/onnxruntime.dll"
		}
	}
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			return "../third_party/onnxruntime_arm64.dylib"
		}
	}
	if runtime.GOOS == "linux" {
		if runtime.GOARCH == "arm64" {
			return "../third_party/onnxruntime_arm64.so"
		}
		return "../third_party/onnxruntime.so"
	}
	panic("Unable to find a version of the onnxruntime library supporting this system.")
}

func initSession() (*ModelSession, error) {
	ort.SetSharedLibraryPath(getSharedLibPath())
	err := ort.InitializeEnvironment()
	if err != nil {
		return nil, fmt.Errorf("Error initializing ORT environment: %w", err)
	}

	inputShape := ort.NewShape(1, 3, 640, 640)
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("Error creating input tensor: %w", err)
	}
	outputShape := ort.NewShape(1, 84, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("Error creating output tensor: %w", err)
	}
	options, err := ort.NewSessionOptions()
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("Error creating ORT session options: %w", err)
	}
	defer options.Destroy()

	// If CoreML is enabled, append the CoreML execution provider
	if useCoreML {
		err = options.AppendExecutionProviderCoreML(0)
		if err != nil {
			inputTensor.Destroy()
			outputTensor.Destroy()
			return nil, fmt.Errorf("Error enabling CoreML: %w", err)
		}
	}

	session, err := ort.NewAdvancedSession(modelPath,
		[]string{"images"}, []string{"output0"},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
		options)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("Error creating ORT session: %w", err)
	}

	return &ModelSession{
		Session: session,
		Input:   inputTensor,
		Output:  outputTensor,
	}, nil
}

func (m *ModelSession) Destroy() {
	m.Session.Destroy()
	m.Input.Destroy()
	m.Output.Destroy()
}

type boundingBox struct {
	label          string
	confidence     float32
	x1, y1, x2, y2 float32
}

func (b *boundingBox) String() string {
	return fmt.Sprintf("Object %s (confidence %f): (%f, %f), (%f, %f)",
		b.label, b.confidence, b.x1, b.y1, b.x2, b.y2)
}

// This loses precision, but recall that the boundingBox has already been
// scaled up to the original image's dimensions. So, it will only lose
// fractional pixels around the edges.
func (b *boundingBox) toRect() image.Rectangle {
	return image.Rect(int(b.x1), int(b.y1), int(b.x2), int(b.y2)).Canon()
}

// Returns the area of b in pixels, after converting to an image.Rectangle.
func (b *boundingBox) rectArea() int {
	size := b.toRect().Size()
	return size.X * size.Y
}

func (b *boundingBox) intersection(other *boundingBox) float32 {
	r1 := b.toRect()
	r2 := other.toRect()
	intersected := r1.Intersect(r2).Canon().Size()
	return float32(intersected.X * intersected.Y)
}

func (b *boundingBox) union(other *boundingBox) float32 {
	intersectArea := b.intersection(other)
	totalArea := float32(b.rectArea() + other.rectArea())
	return totalArea - intersectArea
}

// This won't be entirely precise due to conversion to the integral rectangles
// from the image.Image library, but we're only using it to estimate which
// boxes are overlapping too much, so some imprecision should be OK.
func (b *boundingBox) iou(other *boundingBox) float32 {
	return b.intersection(other) / b.union(other)
}

func processOutput(output []float32, originalWidth,
	originalHeight int) []boundingBox {
	boundingBoxes := make([]boundingBox, 0, 8400)

	var classID int
	var probability float32

	// Iterate through the output array, considering 8400 indices
	for idx := 0; idx < 8400; idx++ {
		// Iterate through 80 classes and find the class with the highest probability
		probability = -1e9
		for col := 0; col < 80; col++ {
			currentProb := output[8400*(col+4)+idx]
			if currentProb > probability {
				probability = currentProb
				classID = col
			}
		}

		// If the probability is less than 0.5, continue to the next index
		if probability < 0.5 {
			continue
		}

		// Extract the coordinates and dimensions of the bounding box
		xc, yc := output[idx], output[8400+idx]
		w, h := output[2*8400+idx], output[3*8400+idx]
		x1 := (xc - w/2) / 640 * float32(originalWidth)
		y1 := (yc - h/2) / 640 * float32(originalHeight)
		x2 := (xc + w/2) / 640 * float32(originalWidth)
		y2 := (yc + h/2) / 640 * float32(originalHeight)

		// Append the bounding box to the result
		boundingBoxes = append(boundingBoxes, boundingBox{
			label:      yoloClasses[classID],
			confidence: probability,
			x1:         x1,
			y1:         y1,
			x2:         x2,
			y2:         y2,
		})
	}

	// Sort the bounding boxes by probability
	sort.Slice(boundingBoxes, func(i, j int) bool {
		return boundingBoxes[i].confidence < boundingBoxes[j].confidence
	})

	// Define a slice to hold the final result
	mergedResults := make([]boundingBox, 0, len(boundingBoxes))

	// Iterate through sorted bounding boxes, removing overlaps
	for _, candidateBox := range boundingBoxes {
		overlapsExistingBox := false
		for _, existingBox := range mergedResults {
			if (&candidateBox).iou(&existingBox) > 0.7 {
				overlapsExistingBox = true
				break
			}
		}
		if !overlapsExistingBox {
			mergedResults = append(mergedResults, candidateBox)
		}
	}

	// This will still be in sorted order by confidence
	return mergedResults
}

// Array of YOLOv8 class labels
var yoloClasses = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
	"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
	"suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
	"bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
	"cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
	"clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}
