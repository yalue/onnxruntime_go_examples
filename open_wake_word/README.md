`onnxruntime_go`: openWakeWord Example
======================================

This example uses the pre-trained models from
[openWakeWord](https://github.com/dscripka/openWakeWord) to check whether a
short `.wav` file contains the wake word "hey jarvis".

openWakeWord is notable for being a pipeline of three separate networks, each
feeding into the next:

 1. `melspectrogram.onnx` converts raw audio into a melspectrogram.
 2. `embedding_model.onnx` converts windows of the melspectrogram into feature
    vectors. This network, along with the melspectrogram, is shared by every
    openWakeWord model.
 3. `hey_jarvis_v0.1.onnx` takes the most recent 16 feature vectors (about two
    seconds of audio) and outputs a score between 0 and 1 indicating whether
    "hey jarvis" was spoken.

The first two networks have inputs and outputs with variable sizes, so this
example runs them using a `DynamicAdvancedSession`, letting `onnxruntime`
allocate the output tensors. The last network has fixed-size inputs and
outputs, so it uses a normal `AdvancedSession`, reusing the same tensors for
every window of audio.

The included `.onnx` files were obtained from the
[openWakeWord v0.5.1 release](https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1).

License
-------

While the openWakeWord code is licensed under Apache 2.0, the pre-trained
openWakeWord models included in this directory are licensed under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
license. They were created by David Scripka; see the
[openWakeWord repository](https://github.com/dscripka/openWakeWord) for more
information.

Example Usage
-------------

Run the program with `-help` to see all command-line flags. In general, you
will need to supply it with an input `.wav` file.

```bash
./open_wake_word -wav_path ./hey_jarvis.wav
./open_wake_word -wav_path ./not_jarvis.wav

# You can adjust the score needed to consider the wake word detected.
./open_wake_word -wav_path ./hey_jarvis.wav -threshold 0.8
```

For example:
```
$ > ./open_wake_word -wav_path ./hey_jarvis.wav
Loaded 3.00 seconds of audio from ./hey_jarvis.wav.
Computed 297 melspectrogram frames, 28 embeddings, and 13 wake word scores.
Scores:
  1.96s: 0.997603
  2.04s: 0.997055
  2.12s: 0.993702
  2.20s: 0.006514
  ...
Max score: 0.997603 at 1.96s
Wake word "hey jarvis" DETECTED in ./hey_jarvis.wav.
Everything seemed to run OK!
```

Using Your Own Audio
--------------------

The input must be a `.wav` file containing 16 kHz, 16-bit, mono PCM audio.
Clips shorter than two seconds will be padded with silence. On Linux, you can
record a three-second clip in the correct format using:

```bash
arecord -f S16_LE -r 16000 -c 1 -d 3 my_clip.wav
```

Or convert an existing audio file using `ffmpeg`:

```bash
ffmpeg -i input.m4a -ar 16000 -ac 1 -sample_fmt s16 my_clip.wav
```
