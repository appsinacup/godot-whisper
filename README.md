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
        <img src="https://img.shields.io/discord/1138836561102897172?logo=discord"
            alt="Chat on Discord"></a>
    <a href="https://whisper.appsinacup.com" alt="Docs">
        Documentation</a>
    <a href="https://discord.gg/v649emcpAu">
</p>

<p align="center">
<img src="whisper_cpp.gif"/>
</p>

## Features

- **Realtime audio transcription**
- **Offline audio transcription**
- **GPU acceleration**
- **Flash Attention**
- **Voice Activity Detection (VAD)**
- **Quantized models**
- **99 languages**
- **Model downloader**

Silero VAD is a neural network that detects speech segments and auto-strips silence during transcription, **preventing hallucinations** from silent audio.

Flash Attention reduces memory usage and improves inference speed, especially for longer audio. Enabled by default — disable via the `flash_attn` property on the `SpeechToText` node if needed.

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

[![Godot Whisper](https://img.youtube.com/vi/fAgjNkfBOKs/0.jpg)](https://www.youtube.com/watch?v=fAgjNkfBOKs&t=10s)

### Multilingual Models

Models manual download link: [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp/tree/main).

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

## Supported Languages

The plugin supports **99 languages** via the `Language` enum on the `SpeechToText` node. Set to `Auto` for automatic language detection (multilingual models only).

For best results with English, use English-only models (`.en` suffix). For other languages, use multilingual models and set the `language` property on the `SpeechToText` node.

## How to install

Go to a github release, copy paste the addons folder to the samples folder. Restart godot editor.

# How to build

# How to build

## Requirements

- Sconstruct(if you want to build locally)
- A language model, can be downloaded in godot editor.

## Global settings

Go to Project -> Project Settings -> General -> Audio -> Input (Check Advance Settings).

You will see a bunch of settings there.

Also, as doing microphone transcribing requires the data to be at a 16000 sampling rate, you can change the audio driver mix rate to 16000: `audio/driver/mix_rate`. This way the resampling won't need to do any work, winning you some valuable 50-100ms for larger audio, but at the price of audio quality.
