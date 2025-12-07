// This is a command-line application that uses an onnx network to convert an
// input string into upper and lower case. The comments here focus on string
// tensor usage; the boilerplate is largely shared with the sum_and_difference
// example, so refer to sum_and_difference for information about onnxruntime_go
// usage in general.
package main

import (
	"flag"
	"fmt"
	ort "github.com/yalue/onnxruntime_go"
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
		if runtime.GOARCH == "amd64" {
			return "../third_party/onnxruntime_amd64.dylib"
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

// Takes a path to the onnxruntime shared library as well as the string that
// will be used as an input to the network. If the network runs successfully,
// it will convert the string to upper and lowercase, and print the results to
// stdout.
func printUpperAndLowercase(onnxruntimeLibPath, inputString string) error {
	ort.SetSharedLibraryPath(onnxruntimeLibPath)
	e := ort.InitializeEnvironment()
	if e != nil {
		return fmt.Errorf("Error initializing the onnxruntime library: %w", e)
	}
	defer ort.DestroyEnvironment()

	// Create a string tensor with the shape [1], to hold our single input
	// string. After creation, this tensor will contain empty strings, so we
	// will set its contents next.
	inputTensor, e := ort.NewStringTensor(ort.NewShape(1))
	if e != nil {
		return fmt.Errorf("Error creating input tensor: %w", e)
	}
	defer inputTensor.Destroy()

	// The input tensor only contains a single string, so it's a bit easier to
	// set its contents using SetElement to set the string at index 0. If the
	// input contained more than one string, we may want to use
	// inputTensor.SetContents(...) instead, which takes an an entire slice of
	// strings at once and should be more efficient when initializing a larger
	// tensor.
	e = inputTensor.SetElement(0, inputString)
	if e != nil {
		return fmt.Errorf("Error setting input tensor contents: %w", e)
	}

	// The network produces two outputs, each with the same dimensions as the
	// input. So, we'll create them now. (We don't need to initialize their
	// contents, just create the string tensors themselves.)
	outputUpper, e := ort.NewStringTensor(ort.NewShape(1))
	if e != nil {
		return fmt.Errorf("Error creating uppercase output tensor: %w", e)
	}
	defer outputUpper.Destroy()
	outputLower, e := ort.NewStringTensor(ort.NewShape(1))
	if e != nil {
		return fmt.Errorf("Error creating lowercase output tensor: %w", e)
	}
	defer outputLower.Destroy()

	// You can refer to the python script to see the input and output names.
	// We just run the session the way we'd run any other session with
	// onnxruntime_go, except onnxruntime populates the output strings.
	onnxPath := "./example_strings.onnx"
	session, e := ort.NewAdvancedSession(onnxPath,
		[]string{"input"}, []string{"output_upper", "output_lower"},
		[]ort.Value{inputTensor}, []ort.Value{outputUpper, outputLower},
		nil)
	if e != nil {
		return fmt.Errorf("Error creating session for %s: %w", onnxPath, e)
	}
	defer session.Destroy()
	e = session.Run()
	if e != nil {
		return fmt.Errorf("Error running %s: %w", onnxPath, e)
	}

	// Unlike with other tensors in onnxruntime_go, the contents of string
	// tensors can't be modified by Go, and obtaining the contents always
	// returns a copy. We use outputTensor.GetElement(...) here since our
	// outputs only contain one string each, but if we had larger output
	// tensors we could use outputTensor.GetContents() instead to get all
	// strings in a slice.
	uppercaseString, e := outputUpper.GetElement(0)
	if e != nil {
		return fmt.Errorf("Error getting uppercase string: %w", e)
	}
	lowercaseString, e := outputLower.GetElement(0)
	if e != nil {
		return fmt.Errorf("Error getting lowercase string: %w", e)
	}

	fmt.Printf("Everything ran OK.\n")
	fmt.Printf("Original input: %s\n", inputString)
	fmt.Printf("Converted to uppercase: %s\n", uppercaseString)
	fmt.Printf("Converted to lowercase: %s\n", lowercaseString)

	return nil
}

func run() int {
	var onnxruntimeLibPath string
	var inputString string
	flag.StringVar(&onnxruntimeLibPath, "onnxruntime_lib",
		getDefaultSharedLibPath(),
		"The path to the onnxruntime shared library for your system.")
	flag.StringVar(&inputString, "input_string", "",
		"The string to convert to upper or lowercase.")
	flag.Parse()
	if onnxruntimeLibPath == "" {
		fmt.Println("You must specify a path to the onnxruntime shared " +
			"on your system. Run with -help for more information.")
		return 1
	}
	if inputString == "" {
		fmt.Println("You must specify an input string. Run with -help for " +
			"more information.")
		return 1
	}
	e := printUpperAndLowercase(onnxruntimeLibPath, inputString)
	if e != nil {
		fmt.Printf("Error running network: %s\n", e)
		return 1
	}
	return 0
}

func main() {
	os.Exit(run())
}
