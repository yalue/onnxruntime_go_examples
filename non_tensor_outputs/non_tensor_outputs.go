// This example illustrates how to access the contents of Maps and sequences,
// using the random-forest sklearn network originally copied and modified from
// here: http://onnx.ai/sklearn-onnx/.
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

func run() int {
	var onnxruntimeLibPath string
	flag.StringVar(&onnxruntimeLibPath, "onnxruntime_lib",
		getDefaultSharedLibPath(),
		"The path to the onnxruntime shared library for your system.")
	flag.Parse()
	if onnxruntimeLibPath == "" {
		fmt.Println("You must specify a path to the onnxruntime shared " +
			"on your system. Run with -help for more information.")
		return 1
	}
	e := runSklearnNetwork(onnxruntimeLibPath)
	if e != nil {
		fmt.Printf("Encountered an error running the network: %s\n", e)
		return 1
	}
	return 0
}

func main() {
	os.Exit(run())
}

func runSklearnNetwork(sharedLibPath string) error {
	ort.SetSharedLibraryPath(sharedLibPath)
	e := ort.InitializeEnvironment()
	if e != nil {
		return fmt.Errorf("Error initializing onnxruntime library: %w", e)
	}

	// Load the session. We'll use DynamicAdvancedSession so that onnxruntime
	// can automatically allocate the more complicated outputs for us.
	modelPath := "./sklearn_randomforest.onnx"
	inputNames := []string{"X"}
	outputNames := []string{"output_label", "output_probability"}
	session, e := ort.NewDynamicAdvancedSession(modelPath, inputNames,
		outputNames, nil)
	if e != nil {
		return fmt.Errorf("Error loading %s: %w", modelPath, e)
	}
	defer session.Destroy()

	// Create the 6x4 input tensor (6 vectors of 4 elements each). This data
	// is from information printed by generate_sklearn_network.py.
	inputShape := ort.NewShape(6, 4)
	inputValues := []float32{
		5.9, 3.0, 5.1, 1.8,
		6.8, 2.8, 4.8, 1.4,
		6.3, 2.3, 4.4, 1.3,
		6.5, 3.0, 5.5, 1.8,
		7.7, 2.8, 6.7, 2.0,
		5.5, 2.5, 4.0, 1.3,
	}
	inputTensor, e := ort.NewTensor(inputShape, inputValues)
	if e != nil {
		return fmt.Errorf("Error creating input tensor: %w", e)
	}
	defer inputTensor.Destroy()

	// Create a two-element slice that will be populated by the values
	// automatically allocated while running the network. (Leaving the outputs
	// as nil allows DynamicAdvancedSession.Run() to allocate them.)
	outputValues := []ort.Value{nil, nil}

	// Actually run the network.
	e = session.Run([]ort.Value{inputTensor}, outputValues)
	if e != nil {
		return fmt.Errorf("Error running %s: %w", modelPath, e)
	}
	// Any auto-allocated outputs must be manually destroyed when no longer
	// needed.
	defer outputValues[0].Destroy()
	defer outputValues[1].Destroy()
	fmt.Printf("Successfully ran %s!\n", modelPath)

	// The first output of this network is just a Tensor containing the labels
	// with the highest probabilities.
	labelTensor := outputValues[0].(*ort.Tensor[int64])
	predictedLabels := labelTensor.GetData()
	for i, v := range predictedLabels {
		fmt.Printf("Predicted label for input %d: %d\n", i, v)
	}

	// The second output of this network is an ONNX Sequence of maps. The
	// sequence contains one map for each of the 6 input vectors. Each map
	// maps every possible label to its predicted probability for the
	// corresponding input vector. (You'll see that the label with the highest
	// probability was already provided in the first output.)
	sequence := outputValues[1].(*ort.Sequence)
	probabilityMaps, e := sequence.GetValues()
	if e != nil {
		return fmt.Errorf("Error getting contents of sequence: %w", e)
	}

	for i := range probabilityMaps {
		// An ONNX Map is represented by two tensors of the same size: one
		// containing keys and one containing values. keys.GetData()[i]
		// contains the key, and values.GetData()[i] contains the value the
		// key maps to.
		m := probabilityMaps[i].(*ort.Map)
		keys, values, e := m.GetKeysAndValues()
		if e != nil {
			return fmt.Errorf("Error getting keys and values for map at "+
				"index %d: %w", i, e)
		}
		keysTensor := keys.(*ort.Tensor[int64])
		valuesTensor := values.(*ort.Tensor[float32])

		fmt.Printf("Individual probabilities for input %d:\n", i)
		for j, key := range keysTensor.GetData() {
			value := valuesTensor.GetData()[j]
			fmt.Printf("   Label %d: %f\n", key, value)
		}
	}
	return nil
}
