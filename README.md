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

|**Realtime audio transcription**| **Offline audio transcription**|
|-|-|
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
| **Web** | CPU (WebGPU disabled until Godot supports it) |

## Video Tutorial

[![Godot Whisper](https://img.youtube.com/vi/fAgjNkfBOKs/0.jpg)](https://www.youtube.com/watch?v=fAgjNkfBOKs&t=10s)

## How to install

### GitHub Release

Go to a [Github Release](https://github.com/appsinacup/godot-whisper/releases), copy paste the addons folder to the samples folder.

### Godot Asset Store

Download directly from [Godot Asset Store](https://store.godotengine.org/asset/appsinacup/godot-whisper-speech-to-text-stt-offline/).

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

Microphone transcription feeds Whisper at 16000 Hz. The addon resamples captured audio from the actual runtime mix rate reported by `AudioServer.get_mix_rate()`.

Optional: set Project Settings -> Audio -> Driver -> Mix Rate (`audio/driver/mix_rate`) to 16000 to avoid resampling overhead. This may reduce overall game audio quality, so only use it if speech transcription is the main audio workload. Godot may still use a different runtime mix rate on some platforms or devices; verify with `AudioServer.get_mix_rate()`. If the runtime mix rate is not 16000, the addon will resample.

## Star History

<a href="https://www.star-history.com/?repos=appsinacup%2Fgodot-whisper&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=appsinacup/godot-whisper&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=appsinacup/godot-whisper&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=appsinacup/godot-whisper&type=date&legend=top-left" />
 </picture>
</a>
