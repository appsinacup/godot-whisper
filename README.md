<p align="center">
	<img width="128px" src="whisper_logo.png"/> 
	<h1 align="center">Godot Whisper</h1> 
</p>

<p align="center">
	<a href="https://github.com/appsinacup/godot-whisper/actions/workflows/runner.yml">
        <img src="https://github.com/appsinacup/godot-whisper/actions/workflows/runner.yml/badge.svg?branch=main"
            alt="build"></a>
    <a href="https://whisper.appsinacup.com" alt="Docs">
        <img src="https://img.shields.io/badge/Documentation-link-%23478cbf?logoColor=white" /></a>
    <a href="https://github.com/ggml-org/whisper.cpp" alt="Whisper CPP">
        <img src="https://img.shields.io/badge/WhisperCPP-v1.8.4-%23478cbf?logoColor=white" /></a>
    <a href="https://github.com/godotengine/godot-cpp" alt="Godot Version">
        <img src="https://img.shields.io/badge/Godot-v4.2-%23478cbf?logo=godot-engine&logoColor=white" /></a>
    <a href="https://discord.gg/v649emcpAu">
        <img src="https://img.shields.io/discord/1138836561102897172?logo=discord"
            alt="Chat on Discord"></a>
</p>

<p align="center">
<img src="whisper_cpp.gif"/>
</p>

## Features

|||
|-|-|
|**Realtime audio transcription**| **Offline audio transcription**|
|**GPU acceleration**| **Flash Attention**|
|**Voice Activity Detection (VAD)**| **Quantized models**|
|**99 languages**| **Model downloader**|

## Platforms

| Platform | GPU Backend |
|----------|-------------|
| **macOS** | Metal + Accelerate |
| **iOS** | Metal + Accelerate |
| **Windows** | OpenCL + Vulkan |
| **Linux** | OpenCL + Vulkan |
| **Android** | OpenCL |
| **Web** | WebGPU |

## Video Tutorial

[![Godot Whisper](https://img.youtube.com/vi/fAgjNkfBOKs/0.jpg)](https://www.youtube.com/watch?v=fAgjNkfBOKs&t=10s)

## How to install

### GitHub Release

Go to a [Github Release](https://github.com/appsinacup/godot-whisper/releases), copy paste the addons folder to the samples folder.

### Godot Assets

Download directly from [Godot Asset Library](https://godotengine.org/asset-library/asset/2638).

**Afterwards**:

Activate the extension in Project -> Project Settings -> Godot Whisper. Restart the Godot editor.

### Models

Models manual download link: [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp/tree/main).

| Model | Size |
|-------|--------|
| **tiny** | 78 MB |
| **base** | 148 MB |
| **small** | 244M |
| **medium** | 769M |
| **large-v1** | 1550M |
| **large-v2** | 1550M |
| **large-v3** | 1550M |
| **large-v3-turbo** | 809M |

## Global settings

Go to Project -> Project Settings -> General -> Audio -> Input (Check Advance Settings).

You will see a bunch of settings there.

Also, as doing microphone transcribing requires the data to be at a 16000 sampling rate, you can change the audio driver mix rate to 16000: `audio/driver/mix_rate`. This way the resampling won't need to do any work, winning you some valuable 50-100ms for larger audio, but at the price of audio quality.
