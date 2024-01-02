// This is a command-line application that should behave identically to the
// plain "mnist" example, but using float16 types. A large amount of this
// program was simply copied from the mnist example, but modified to use
// 16-bit floats with the help of github.com/x448/float16.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"github.com/x448/float16"
	ort "github.com/yalue/onnxruntime_go"
	"image"
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"os"
	"runtime"
)

// For more comments, see the sum_and_difference example.
func getDefaultSharedLibPath() string {
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
	fmt.Printf("Unable to determine a path to the onnxruntime shared library"+
		" for OS \"%s\" and architecture \"%s\".\n", runtime.GOOS,
		runtime.GOARCH)
	return ""
}

// Implements the color interface
type grayscaleFloat float32

func (f grayscaleFloat) RGBA() (r, g, b, a uint32) {
	a = 0xffff
	v := uint32(f * 0xffff)
	if v > 0xffff {
		v = 0xffff
	}
	r = v
	g = v
	b = v
	return
}

// Used to satisfy the image interface as well as to help with formatting and
// resizing an input image into the format expected as a network input.
type ProcessedImage struct {
	// The number of "pixels" in the input image corresponding to a single
	// pixel in the 28x28 output image.
	dx, dy float32

	// The input image being transformed
	pic image.Image

	// If true, the grayscale values in the postprocessed image will be
	// inverted, so that dark colors in the original become light, and vice
	// versa. Recall that the network expects black backgrounds, so this should
	// be set to true for images with light backgrounds.
	Invert bool
}

func (p *ProcessedImage) ColorModel() color.Model {
	return color.Gray16Model
}

func (p *ProcessedImage) Bounds() image.Rectangle {
	return image.Rect(0, 0, 28, 28)
}

// Returns an average grayscale value using the pixels in the input image.
func (p *ProcessedImage) At(x, y int) color.Color {
	if (x < 0) || (x >= 28) || (y < 0) || (y >= 28) {
		return color.Black
	}

	// Compute the window of pixels in the input image we'll be averaging.
	startX := int(float32(x) * p.dx)
	endX := int(float32(x+1) * p.dx)
	if endX == startX {
		endX = startX + 1
	}
	startY := int(float32(y) * p.dy)
	endY := int(float32(y+1) * p.dy)
	if endY == startY {
		endY = startY + 1
	}

	// Compute the average brightness over the window of pixels
	var sum float32
	var nPix int
	for row := startY; row < endY; row++ {
		for col := startX; col < endX; col++ {
			c := p.pic.At(col, row)
			grayValue := color.Gray16Model.Convert(c).(color.Gray16).Y
			sum += float32(grayValue) / 0xffff
			nPix++
		}
	}

	brightness := grayscaleFloat(sum / float32(nPix))
	if p.Invert {
		brightness = 1.0 - brightness
	}
	return brightness
}

// Returns the float16 network inputs as a slice of bytes to be used with a
// CustomDataTensor. This is where we convert the float32 grayscale image to
// float16.
func (p *ProcessedImage) GetNetworkInput() []byte {
	// We need two bytes per float16 pixel, and 28x28 to form the image.
	toReturn := make([]byte, 28*28*2)
	currentOffset := 0
	for row := 0; row < 28; row++ {
		for col := 0; col < 28; col++ {
			c := float32(p.At(col, row).(grayscaleFloat))
			valueFloat16 := float16.Fromfloat32(c)
			// The float16.Float16 type is just a uint16 underneath; write its
			// bytes to the data slice.
			binary.LittleEndian.PutUint16(toReturn[currentOffset:],
				uint16(valueFloat16))
			currentOffset += 2
		}
	}
	return toReturn
}

// Takes a path to an image file, loads the image, and returns a ProcessedImage
// struct which can be used to obtain the neural network input.
func NewProcessedImage(path string, invertBrightness bool) (*ProcessedImage,
	error) {
	f, e := os.Open(path)
	if e != nil {
		return nil, fmt.Errorf("Error opening %s: %w", path, e)
	}
	defer f.Close()
	originalPic, _, e := image.Decode(f)
	if e != nil {
		return nil, fmt.Errorf("Error decoding image %s: %w", path, e)
	}
	bounds := originalPic.Bounds().Canon()
	if (bounds.Min.X != 0) || (bounds.Min.Y != 0) {
		// Should never happen with the standard library.
		return nil, fmt.Errorf("Bounding rect of %s doesn't start at 0, 0",
			path)
	}
	return &ProcessedImage{
		dx:     float32(bounds.Dx()) / 28.0,
		dy:     float32(bounds.Dy()) / 28.0,
		pic:    originalPic,
		Invert: invertBrightness,
	}, nil
}

// Attempts to save the given image as a png.
func saveImage(pic image.Image, path string) error {
	f, e := os.Create(path)
	if e != nil {
		return fmt.Errorf("Error creating %s: %w", path, e)
	}
	defer f.Close()
	e = png.Encode(f, pic)
	if e != nil {
		return fmt.Errorf("Error encoding PNG image to %s: %w", path, e)
	}
	return nil
}

// Takes a list of float16 values as a slice of bytes, and converts each
// float16 to a float32.
func convertFloat16Data(data []byte) ([]float32, error) {
	if (len(data) % 2) != 0 {
		return nil, fmt.Errorf("A slice of float16s must have an even length")
	}
	toReturn := make([]float32, len(data)/2)
	for i := range toReturn {
		valueUint16 := binary.LittleEndian.Uint16(data[i*2:])
		valueFloat16 := float16.Frombits(valueUint16)
		toReturn[i] = valueFloat16.Float32()
	}
	return toReturn, nil
}

// Takes a path to the onnxruntime shared library as well as the image file
// containing a digit to be classified. The image file will be processed into
// the format expected by the .onnx network.
//
// If the network runs successfully, this will print the classification results
// to stdout.
func classifyDigit(onnxruntimeLibPath, imagePath string,
	invertBrightness bool) error {
	ort.SetSharedLibraryPath(onnxruntimeLibPath)
	e := ort.InitializeEnvironment()
	if e != nil {
		return fmt.Errorf("Error initializing the onnxruntime library: %w", e)
	}
	defer ort.DestroyEnvironment()

	// Load the input image and save the postprocessed version for a visual
	// inspection.
	inputImage, e := NewProcessedImage(imagePath, invertBrightness)
	if e != nil {
		return fmt.Errorf("Error loading input image: %w", e)
	}
	postprocessedPath := "./postprocessed_input_image.png"
	e = saveImage(inputImage, postprocessedPath)
	if e != nil {
		fmt.Printf("Error saving postprocessed input: %s. Continuing.\n", e)
	} else {
		fmt.Printf("Saved postprocessed input image to %s.\n",
			postprocessedPath)
	}

	// Create and populate the input tensor
	inputShape := ort.NewShape(1, 1, 28, 28)
	inputData := inputImage.GetNetworkInput()
	input, e := ort.NewCustomDataTensor(inputShape, inputData,
		ort.TensorElementDataTypeFloat16)
	if e != nil {
		return fmt.Errorf("Error creating input tensor: %w", e)
	}
	defer input.Destroy()

	// Create the output tensor. We need a 1x10 float16 tensor, with two bytes
	// per float16.
	outputShape := ort.NewShape(1, 10)
	outputData := make([]byte, outputShape.FlattenedSize()*2)
	output, e := ort.NewCustomDataTensor(outputShape, outputData,
		ort.TensorElementDataTypeFloat16)
	if e != nil {
		return fmt.Errorf("Error creating output tensor: %w", e)
	}
	defer output.Destroy()

	// The input and output names are required by this network; they can be
	// found on the MNIST ONNX models page linked in the README.
	session, e := ort.NewAdvancedSession("./mnist_float16.onnx",
		[]string{"Input3"}, []string{"Plus214_Output_0"},
		[]ort.ArbitraryTensor{input}, []ort.ArbitraryTensor{output}, nil)
	if e != nil {
		return fmt.Errorf("Error creating MNIST network session: %w", e)
	}
	defer session.Destroy()

	// Run the network and print the results.
	e = session.Run()
	if e != nil {
		return fmt.Errorf("Error running the MNIST network: %w", e)
	}

	// Convert the outputs from float16 back to float32 to make them easier to
	// compare and print.
	outputFloat32, e := convertFloat16Data(output.GetData())
	if e != nil {
		return fmt.Errorf("Error converting float16 bytes to float32's: %w", e)
	}

	// Find the most likely output.
	maxIndex := 0
	maxProbability := float32(-1.0e9)
	for i, v := range outputFloat32 {
		fmt.Printf("  %d: %f\n", i, v)
		if v > maxProbability {
			maxProbability = v
			maxIndex = i
		}
	}
	fmt.Printf("%s is probably a %d, with probability %f\n", imagePath,
		maxIndex, maxProbability)

	return nil
}

func run() int {
	var onnxruntimeLibPath string
	var imagePath string
	var invertImage bool
	flag.StringVar(&onnxruntimeLibPath, "onnxruntime_lib",
		getDefaultSharedLibPath(),
		"The path to the onnxruntime shared library for your system.")
	flag.StringVar(&imagePath, "image_path", "",
		"The image containing a digit to classify.")
	flag.BoolVar(&invertImage, "invert_image", false,
		"If set, the image's colors will be inverted before processing. "+
			"The network expects inputs with dark backgrounds, so you should "+
			"set this to true for images with light backgrounds.")
	flag.Parse()
	if onnxruntimeLibPath == "" {
		fmt.Println("You must specify a path to the onnxruntime shared " +
			"on your system. Run with -help for more information.")
		return 1
	}
	if imagePath == "" {
		fmt.Println("You must specify an input image. Run with -help for " +
			"more information.")
		return 1
	}
	e := classifyDigit(onnxruntimeLibPath, imagePath, invertImage)
	if e != nil {
		fmt.Printf("Error running network: %s\n", e)
		return 1
	}
	fmt.Printf("Everything seemed to run OK!\n")
	return 0
}

func main() {
	os.Exit(run())
}
