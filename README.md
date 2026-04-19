<p align="center">
	<img width="128px" src="whisper_logo.png"/> 
	<h1 align="center">Godot Whisper</h1> 
</p>

<p align="center">
	<a href="https://github.com/appsinacup/godot-whisper/actions/workflows/runner.yml">
        <img src="https://github.com/appsinacup/godot-whisper/actions/workflows/runner.yml/badge.svg?branch=main"
            alt="chat on Discord"></a>
    <a href="https://github.com/ggml-org/whisper.cpp" alt="Whisper CPP">
        <img src="https://img.shields.io/badge/WhisperCPP-v1.8.4-%23478cbf?logoColor=white" /></a>
    <a href="https://github.com/godotengine/godot-cpp" alt="Godot Version">
        <img src="https://img.shields.io/badge/Godot-v4.2-%23478cbf?logo=godot-engine&logoColor=white" /></a>
    <a href="https://github.com/appsinacup/godot-whisper/graphs/contributors" alt="Contributors">
        <img src="https://img.shields.io/github/contributors/appsinacup/godot-whisper" /></a>
    <a href="https://github.com/appsinacup/godot-whisper/pulse" alt="Activity">
        <img src="https://img.shields.io/github/commit-activity/m/appsinacup/godot-whisper" /></a>
    <a href="https://discord.gg/v649emcpAu">
        <img src="https://img.shields.io/discord/1138836561102897172?logo=discord"
            alt="Chat on Discord"></a>
</p>

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<p align="center">
<img src="whisper_cpp.gif"/>
</p>

## Features

- **Realtime audio transcription** — microphone capture and live transcription on a separate thread.
- **Offline audio transcription** — transcribe pre-recorded WAV files directly in the editor.
- **GPU acceleration** — Metal (macOS/iOS), OpenCL + Vulkan (Windows/Linux/Android), WebGPU (Web).
- **Flash Attention** — memory-efficient attention enabled by default, configurable via `flash_attn` property.
- **Voice Activity Detection (VAD)** — Silero neural network VAD auto-strips silence and prevents hallucinations.
- **Quantized models** — Q5_0, Q5_1, Q8_0 support for smaller file sizes with minimal quality loss.
- **99 languages** — automatic language detection or manual selection via the `language` property.
- **Model downloader** — download models directly from the Godot editor.

## Platforms

| Platform | GPU Backend | Notes |
|----------|-------------|-------|
| **macOS** | Metal + Accelerate | GPU-accelerated via Metal |
| **iOS** | Metal + Accelerate | GPU-accelerated via Metal |
| **Windows** | OpenCL + Vulkan | x86_32, x86_64, arm64. Vulkan auto-detected when `glslc` is available |
| **Linux** | OpenCL + Vulkan | Vulkan auto-detected when `glslc` is available |
| **Android** | OpenCL | GPU via OpenCL (Vulkan not supported upstream by whisper.cpp) |
| **Web** | WebGPU | `scons webgpu=yes` — **experimental**, requires matching Emscripten/Dawn versions. CPU-only by default |

## Video Tutorial

[![Comparison](https://img.youtube.com/vi/fAgjNkfBOKs/0.jpg)](https://www.youtube.com/watch?v=fAgjNkfBOKs&t=10s)

## Whisper Models

All OpenAI Whisper models are supported. Load the corresponding `.bin` file — no code changes needed. Models can be downloaded directly in the Godot editor or from [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp/tree/main).

### Multilingual Models

| Model | Params | Full (f16) | Q8_0 | Q5_0 / Q5_1 | Best for |
|-------|--------|---:|---:|---:|----------|
| **tiny** | 39M | 78 MB | 44 MB | 32 MB (q5_1) | Prototyping, low-end devices |
| **base** | 74M | 148 MB | 82 MB | 60 MB (q5_1) | Mobile, real-time on most devices |
| **small** | 244M | 488 MB | 264 MB | 190 MB (q5_1) | Good balance of speed and quality |
| **medium** | 769M | 1.53 GB | 823 MB | 539 MB (q5_0) | High-quality transcription |
| **large-v1** | 1550M | 3.09 GB | — | — | First large model |
| **large-v2** | 1550M | 3.09 GB | 1.66 GB | 1.08 GB (q5_0) | Best Whisper v2 |
| **large-v3** | 1550M | 3.10 GB | — | 1.08 GB (q5_0) | Best multilingual accuracy |
| **large-v3-turbo** | 809M | 1.62 GB | 874 MB | 574 MB (q5_0) | ⭐ Recommended — fast + accurate |

### English-only Models

English-only models (`.en` suffix) are faster and more accurate for English. Available for tiny through medium:

| Model | Full (f16) | Q8_0 | Q5_1 |
|-------|---:|---:|---:|
| **tiny.en** | 78 MB | 44 MB | 32 MB |
| **base.en** | 148 MB | 82 MB | 60 MB |
| **small.en** | 488 MB | 264 MB | 190 MB |
| **medium.en** | 1.53 GB | 823 MB | 539 MB (q5_0) |

### Quantized Models

Quantized models store weights with lower precision, reducing file size with minimal quality loss:

| Format | Bits | Size Reduction |
|--------|------|---------------|
| Q8_0 | 8 | ~50% smaller |
| Q5_1 | 5 | ~65% smaller |
| Q5_0 | 5 | ~65% smaller |

Available for all model sizes (both multilingual and English-only). Pre-quantized `.bin` files are on [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp/tree/main).

**Choosing a model:**
- **English-only** → use `.en` models for better speed and accuracy.
- **Multilingual** → use models without `.en` suffix and set the `language` property.
- **Mobile/Web** → use quantized (Q5_0/Q5_1) to reduce download size and memory.
- **large-v3-turbo** → recommended for most use cases — nearly large-v3 quality at 3× the speed.

## Voice Activity Detection (VAD)

Silero VAD is a neural network that detects speech segments and auto-strips silence during transcription, **preventing hallucinations** from silent audio.

**Setup:**
1. Download `ggml-silero-v6.2.0.bin` (~1.9 MB) — available in the Godot editor model downloader or from [Hugging Face](https://huggingface.co/ggml-org/whisper-vad)
2. Set `vad_model_path` on the `SpeechToText` node
3. Set `enable_vad = true`

When enabled, `transcribe()` automatically filters out silence before processing. You can also detect speech segments manually:

```gdscript
var segments = stt.detect_speech_segments(audio_buffer)
for seg in segments:
    print("Speech from %.2fs to %.2fs" % [seg.start, seg.end])
```

## Flash Attention

Flash Attention reduces memory usage and improves inference speed, especially for longer audio. Enabled by default — disable via the `flash_attn` property on the `SpeechToText` node if needed.

## Supported Languages

The plugin supports **99 languages** via the `Language` enum on the `SpeechToText` node. Set to `Auto` for automatic language detection (multilingual models only).

For best results with English, use English-only models (`.en` suffix). For other languages, use multilingual models and set the `language` property on the `SpeechToText` node.

See the [OpenAI Whisper paper](https://cdn.openai.com/papers/whisper.pdf) for detailed per-language accuracy benchmarks.

## How to install

Go to a github release, copy paste the addons folder to the samples folder. Restart godot editor.

</p>
<p align="center">
<img src="banner_godot_whisper.jpg"/>
</p>

# How to build

## Requirements

- Sconstruct(if you want to build locally)
- A language model, can be downloaded in godot editor.

## AudioStreamToText

`AudioStreamToText` - this node can be used in editor to check transcribing. Simply add a WAV audio source and click start_transcribe button.

Normal times for this, using tiny.en model are about 0.3s. This only does transcribing.

NOTE: Currently this node supports only some .WAV files. The transcribe function takes as input a `PackedFloat32Array` buffer. Currently the only format supported is if the .WAV is `AudioStreamWAV.FORMAT_8_BITS` and `AudioStreamWAV.FORMAT_16_BITS`. For other it will simply not work and you will have to write a custom decoder for the .WAV file. Godot does support decoding it at runtime, check how CaptureStreamToText node works.

## CaptureStreamToText

This runs also resampling on the audio(in case mix rate is not exactly 16000 it will process the audio to 16000). Then it runs every transcribe_interval transcribe function.

## Initial Prompt

For Chinese, if you want to select between Traditional and Simplified, you need to provide an initial prompt with the one you want, and then the model should keep that same one going. See [Whisper Discussion #277](https://github.com/openai/whisper/discussions/277).

Also, if you have problems with punctuation, you can give it an initial prompt with punctuation. See [Whisper Discussion #194](https://github.com/openai/whisper/discussions/194).

## Language Model

Go to any `StreamToText` node, select a Language Model to Download and click Download. You might have to alt tab editor or restart for asset to appear. Then, select `language_model` property.

## Global settings

Go to Project -> Project Settings -> General -> Audio -> Input (Check Advance Settings).

You will see a bunch of settings there.

Also, as doing microphone transcribing requires the data to be at a 16000 sampling rate, you can change the audio driver mix rate to 16000: `audio/driver/mix_rate`. This way the resampling won't need to do any work, winning you some valuable 50-100ms for larger audio, but at the price of audio quality.
