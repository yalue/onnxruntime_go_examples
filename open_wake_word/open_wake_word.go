// This is a command-line application that uses the pre-trained openWakeWord
// models to check whether a short .wav file contains the wake word "hey
// jarvis".
//
// openWakeWord is a pipeline of three separate .onnx networks, each feeding
// the next:
//
//  1. melspectrogram.onnx converts raw 16 kHz audio samples into a
//     melspectrogram with 32 frequency bins per 10 ms frame.
//  2. embedding_model.onnx converts each window of 76 melspectrogram frames
//     into a 96-element feature vector ("embedding"). A new embedding is
//     produced for every 8 frames (80 ms) of audio.
//  3. hey_jarvis_v0.1.onnx looks at the 16 most recent embeddings and outputs
//     a single score between 0 and 1 indicating how likely it is that the
//     wake word was spoken.
//
// The first two networks are shared by all openWakeWord wake word models; only
// the last network is specific to "hey jarvis".
//
// This program shares a fair amount of boilerplate with the simpler
// sum_and_difference example, which includes far more comments and may be an
// easier starting point for someone entirely new to the onnxruntime_go
// library.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	ort "github.com/yalue/onnxruntime_go"
	"io"
	"os"
	"runtime"
)

const (
	// The only sample rate supported by the openWakeWord models.
	sampleRate = 16000

	// The number of frequency bins in each melspectrogram frame.
	melBins = 32

	// The number of melspectrogram frames used to compute one embedding, and
	// the number of frames between the start of consecutive embeddings.
	melWindowSize = 76
	melWindowStep = 8

	// The size of each embedding vector.
	embeddingSize = 96

	// The number of consecutive embeddings the wake word model looks at.
	wakeWordWindowSize = 16

	// The duration of a single melspectrogram frame, in seconds.
	melFrameSeconds = 0.01
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

// Loads a .wav file containing 16 kHz, 16-bit, mono PCM audio. Returns the
// samples converted to float32. Note that the samples are NOT normalized to
// the range [-1, 1]; the melspectrogram network expects values in the
// original int16 range.
//
// This is only a minimal .wav parser, intended to avoid adding dependencies
// to this example. It rejects any other audio format.
func loadWav(path string) ([]float32, error) {
	f, e := os.Open(path)
	if e != nil {
		return nil, fmt.Errorf("Error opening %s: %w", path, e)
	}
	defer f.Close()

	// A .wav file starts with a 12-byte RIFF header, followed by a sequence
	// of chunks. Each chunk has a 4-byte ID and a 4-byte little-endian size.
	var riffHeader [12]byte
	_, e = io.ReadFull(f, riffHeader[:])
	if e != nil {
		return nil, fmt.Errorf("Error reading RIFF header: %w", e)
	}
	if (string(riffHeader[0:4]) != "RIFF") ||
		(string(riffHeader[8:12]) != "WAVE") {
		return nil, fmt.Errorf("%s is not a .wav file", path)
	}

	gotFormat := false
	for {
		var chunkHeader [8]byte
		_, e = io.ReadFull(f, chunkHeader[:])
		if e != nil {
			return nil, fmt.Errorf("Error reading chunk header (no data "+
				"chunk found?): %w", e)
		}
		chunkID := string(chunkHeader[0:4])
		chunkSize := binary.LittleEndian.Uint32(chunkHeader[4:8])

		switch chunkID {
		case "fmt ":
			formatData := make([]byte, chunkSize)
			_, e = io.ReadFull(f, formatData)
			if e != nil {
				return nil, fmt.Errorf("Error reading fmt chunk: %w", e)
			}
			if chunkSize < 16 {
				return nil, fmt.Errorf("Invalid fmt chunk size: %d",
					chunkSize)
			}
			audioFormat := binary.LittleEndian.Uint16(formatData[0:2])
			channels := binary.LittleEndian.Uint16(formatData[2:4])
			rate := binary.LittleEndian.Uint32(formatData[4:8])
			bitsPerSample := binary.LittleEndian.Uint16(formatData[14:16])
			if (audioFormat != 1) || (channels != 1) ||
				(rate != sampleRate) || (bitsPerSample != 16) {
				return nil, fmt.Errorf("Unsupported audio format (format "+
					"%d, %d channels, %d Hz, %d bits per sample). The "+
					"audio must be 16 kHz, 16-bit, mono PCM. See the "+
					"README for how to convert it", audioFormat, channels,
					rate, bitsPerSample)
			}
			gotFormat = true
		case "data":
			if !gotFormat {
				return nil, fmt.Errorf("Found data chunk before fmt chunk")
			}
			rawSamples := make([]int16, chunkSize/2)
			e = binary.Read(f, binary.LittleEndian, rawSamples)
			if e != nil {
				return nil, fmt.Errorf("Error reading audio samples: %w", e)
			}
			toReturn := make([]float32, len(rawSamples))
			for i, v := range rawSamples {
				toReturn[i] = float32(v)
			}
			return toReturn, nil
		default:
			// Skip chunks we don't care about, e.g., the "LIST" metadata
			// chunk written by ffmpeg. Chunks are padded to an even size.
			_, e = f.Seek(int64(chunkSize+(chunkSize&1)), io.SeekCurrent)
			if e != nil {
				return nil, fmt.Errorf("Error skipping %s chunk: %w",
					chunkID, e)
			}
		}
	}
}

// Runs the melspectrogram network on the given audio samples. Returns the
// melspectrogram as a flat slice containing melBins values per frame, along
// with the number of frames.
func computeMelspectrogram(samples []float32) ([]float32, int, error) {
	// The number of output frames depends on the length of the audio, so we
	// use a DynamicAdvancedSession, which lets us provide inputs and outputs
	// with different shapes each time Run() is called.
	session, e := ort.NewDynamicAdvancedSession("./melspectrogram.onnx",
		[]string{"input"}, []string{"output"}, nil)
	if e != nil {
		return nil, 0, fmt.Errorf("Error creating melspectrogram session: %w",
			e)
	}
	defer session.Destroy()

	// The input shape is (batch size, number of samples).
	input, e := ort.NewTensor(ort.NewShape(1, int64(len(samples))), samples)
	if e != nil {
		return nil, 0, fmt.Errorf("Error creating melspectrogram input: %w",
			e)
	}
	defer input.Destroy()

	// Rather than computing the output shape ourselves, we pass a nil output
	// value. This instructs onnxruntime to allocate an output tensor of the
	// correct size, which Run() places into the outputs slice.
	outputs := []ort.Value{nil}
	e = session.Run([]ort.Value{input}, outputs)
	if e != nil {
		return nil, 0, fmt.Errorf("Error running melspectrogram network: %w",
			e)
	}
	defer outputs[0].Destroy()

	// We know this network outputs float32 data, so we can convert the
	// automatically-allocated Value into a *Tensor[float32].
	output, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, 0, fmt.Errorf("Unexpected melspectrogram output type")
	}

	// The output shape is (batch size, 1, number of frames, melBins).
	frameCount := int(output.GetShape()[2])

	// The openWakeWord python code applies this transformation to the
	// network's output, so we must do the same. We also copy the data,
	// because the tensor's memory is freed when it's destroyed.
	outputData := output.GetData()
	mel := make([]float32, len(outputData))
	for i, v := range outputData {
		mel[i] = v/10.0 + 2.0
	}
	return mel, frameCount, nil
}

// Runs the embedding network on every window of melWindowSize frames in the
// given melspectrogram. Returns the embeddings as a flat slice containing
// embeddingSize values per embedding, along with the number of embeddings.
func computeEmbeddings(mel []float32, frameCount int) ([]float32, int,
	error) {
	if frameCount < melWindowSize {
		return nil, 0, fmt.Errorf("The audio is too short to compute any "+
			"embeddings (got %d melspectrogram frames, need %d)", frameCount,
			melWindowSize)
	}
	embeddingCount := (frameCount-melWindowSize)/melWindowStep + 1

	session, e := ort.NewDynamicAdvancedSession("./embedding_model.onnx",
		[]string{"input_1"}, []string{"conv2d_19"}, nil)
	if e != nil {
		return nil, 0, fmt.Errorf("Error creating embedding session: %w", e)
	}
	defer session.Destroy()

	// Rather than running the network once per window, we compute every
	// embedding in a single Run() by stacking all of the windows along the
	// first (batch) dimension. The input shape is (batch size, window size,
	// melBins, 1). Since the windows overlap, each is copied separately.
	windowValues := melWindowSize * melBins
	inputData := make([]float32, embeddingCount*windowValues)
	for i := 0; i < embeddingCount; i++ {
		start := i * melWindowStep * melBins
		copy(inputData[i*windowValues:], mel[start:start+windowValues])
	}
	input, e := ort.NewTensor(ort.NewShape(int64(embeddingCount),
		melWindowSize, melBins, 1), inputData)
	if e != nil {
		return nil, 0, fmt.Errorf("Error creating embedding input: %w", e)
	}
	defer input.Destroy()

	// As with the melspectrogram, let onnxruntime allocate the output. Its
	// shape will be (batch size, 1, 1, embeddingSize).
	outputs := []ort.Value{nil}
	e = session.Run([]ort.Value{input}, outputs)
	if e != nil {
		return nil, 0, fmt.Errorf("Error running embedding network: %w", e)
	}
	defer outputs[0].Destroy()
	output, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, 0, fmt.Errorf("Unexpected embedding output type")
	}
	embeddings := make([]float32, embeddingCount*embeddingSize)
	copy(embeddings, output.GetData())
	return embeddings, embeddingCount, nil
}

// Runs the "hey jarvis" network on every window of wakeWordWindowSize
// consecutive embeddings. Returns the score for each window.
func scoreWakeWord(embeddings []float32, embeddingCount int) ([]float32,
	error) {
	if embeddingCount < wakeWordWindowSize {
		return nil, fmt.Errorf("The audio is too short to run the wake word "+
			"model (got %d embeddings, need %d)", embeddingCount,
			wakeWordWindowSize)
	}

	// Unlike the other two networks, the wake word network has a fixed
	// input shape of (1, wakeWordWindowSize, embeddingSize) and output shape
	// of (1, 1). So, we can use a normal AdvancedSession, creating the input
	// and output tensors once and reusing them for every window. A real-time
	// detector would work the same way, running the network every 80 ms as
	// new audio arrives.
	input, e := ort.NewEmptyTensor[float32](ort.NewShape(1,
		wakeWordWindowSize, embeddingSize))
	if e != nil {
		return nil, fmt.Errorf("Error creating wake word input: %w", e)
	}
	defer input.Destroy()
	output, e := ort.NewEmptyTensor[float32](ort.NewShape(1, 1))
	if e != nil {
		return nil, fmt.Errorf("Error creating wake word output: %w", e)
	}
	defer output.Destroy()

	session, e := ort.NewAdvancedSession("./hey_jarvis_v0.1.onnx",
		[]string{"x.1"}, []string{"53"},
		[]ort.Value{input}, []ort.Value{output}, nil)
	if e != nil {
		return nil, fmt.Errorf("Error creating wake word session: %w", e)
	}
	defer session.Destroy()

	windowCount := embeddingCount - wakeWordWindowSize + 1
	scores := make([]float32, windowCount)
	inputData := input.GetData()
	for i := 0; i < windowCount; i++ {
		// Writing to the slice returned by GetData() changes the tensor's
		// contents directly, so there's no need to create a new tensor.
		start := i * embeddingSize
		copy(inputData, embeddings[start:start+len(inputData)])
		e = session.Run()
		if e != nil {
			return nil, fmt.Errorf("Error running wake word network: %w", e)
		}
		scores[i] = output.GetData()[0]
	}
	return scores, nil
}

// Takes a path to the onnxruntime shared library and to a .wav file. Prints
// whether the wake word was detected in the audio.
func detectWakeWord(onnxruntimeLibPath, wavPath string,
	threshold float32) error {
	ort.SetSharedLibraryPath(onnxruntimeLibPath)
	e := ort.InitializeEnvironment()
	if e != nil {
		return fmt.Errorf("Error initializing the onnxruntime library: %w", e)
	}
	defer ort.DestroyEnvironment()

	samples, e := loadWav(wavPath)
	if e != nil {
		return fmt.Errorf("Error loading %s: %w", wavPath, e)
	}
	fmt.Printf("Loaded %.2f seconds of audio from %s.\n",
		float32(len(samples))/sampleRate, wavPath)

	// The wake word model needs about two seconds of audio to produce a
	// single score, so pad short clips with silence at the beginning.
	minSamples := 2 * sampleRate
	if len(samples) < minSamples {
		padded := make([]float32, minSamples)
		copy(padded[minSamples-len(samples):], samples)
		samples = padded
	}

	mel, frameCount, e := computeMelspectrogram(samples)
	if e != nil {
		return e
	}
	embeddings, embeddingCount, e := computeEmbeddings(mel, frameCount)
	if e != nil {
		return e
	}
	scores, e := scoreWakeWord(embeddings, embeddingCount)
	if e != nil {
		return e
	}
	fmt.Printf("Computed %d melspectrogram frames, %d embeddings, and %d "+
		"wake word scores.\n", frameCount, embeddingCount, len(scores))

	// Print the score for each window, labeled by the time at the end of the
	// window's audio.
	maxIndex := 0
	for i, score := range scores {
		if score > scores[maxIndex] {
			maxIndex = i
		}
	}
	windowEndTime := func(i int) float32 {
		lastEmbedding := i + wakeWordWindowSize - 1
		lastFrame := lastEmbedding*melWindowStep + melWindowSize
		return float32(lastFrame) * melFrameSeconds
	}
	fmt.Printf("Scores:\n")
	for i, score := range scores {
		fmt.Printf("  %.2fs: %f\n", windowEndTime(i), score)
	}

	maxScore := scores[maxIndex]
	fmt.Printf("Max score: %f at %.2fs\n", maxScore, windowEndTime(maxIndex))
	if maxScore >= threshold {
		fmt.Printf("Wake word \"hey jarvis\" DETECTED in %s.\n", wavPath)
	} else {
		fmt.Printf("Wake word \"hey jarvis\" not detected in %s.\n", wavPath)
	}
	return nil
}

func run() int {
	var onnxruntimeLibPath string
	var wavPath string
	var threshold float64
	flag.StringVar(&onnxruntimeLibPath, "onnxruntime_lib",
		getDefaultSharedLibPath(),
		"The path to the onnxruntime shared library for your system.")
	flag.StringVar(&wavPath, "wav_path", "",
		"The .wav file to check for the wake word. Must contain 16 kHz, "+
			"16-bit, mono PCM audio.")
	flag.Float64Var(&threshold, "threshold", 0.5,
		"The score, between 0 and 1, above which the wake word is "+
			"considered detected.")
	flag.Parse()
	if onnxruntimeLibPath == "" {
		fmt.Println("You must specify a path to the onnxruntime shared library " +
			"on your system. Run with -help for more information.")
		return 1
	}
	if wavPath == "" {
		fmt.Println("You must specify an input .wav file. Run with -help " +
			"for more information.")
		return 1
	}
	e := detectWakeWord(onnxruntimeLibPath, wavPath, float32(threshold))
	if e != nil {
		fmt.Printf("Error running wake word detection: %s\n", e)
		return 1
	}
	fmt.Printf("Everything seemed to run OK!\n")
	return 0
}

func main() {
	os.Exit(run())
}
