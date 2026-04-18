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
        <img src="https://img.shields.io/badge/Godot-v4.1-%23478cbf?logo=godot-engine&logoColor=white" /></a>
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

- Realtime audio transcribe.
- Audio transcribe with recorded audio.
- Runs on separate thread.
- Metal for Apple devices.
- Vulkan for Windows/Linux/Android.
- WebGPU for web builds.
- Voice Activity Detection (VAD).
- Flash Attention (enabled by default).

## Platform & Backend Status

| Platform | Backend | Status | Notes |
|----------|---------|--------|-------|
| **macOS** | Metal + Accelerate | ✅ Supported | GPU-accelerated via Metal |
| **iOS** | Metal + Accelerate | ✅ Supported | GPU-accelerated via Metal |
| **Windows** | Vulkan | ✅ Supported | Cross-vendor GPU (AMD, NVIDIA, Intel) |
| **Linux** | Vulkan | ✅ Supported | Cross-vendor GPU (AMD, NVIDIA, Intel) |
| **Android** | Vulkan | ✅ Supported | Mobile GPU acceleration |
| **Web** | WebGPU | ✅ Supported | GPU acceleration in browsers |
| **Web** | CPU fallback | ✅ Supported | WASM, no GPU needed |
| **macOS/iOS** | CoreML | ⬜ Planned | Apple Neural Engine acceleration |
| **Windows/Linux** | CUDA | ⬜ Planned | NVIDIA-specific, faster than Vulkan |
| **Windows/Linux** | OpenCL (CLBlast) | ❌ Removed | Replaced by Vulkan |

### Feature Status

| Feature | Status | Notes |
|---------|--------|-------|
| Realtime transcription | ✅ | Microphone capture + transcribe |
| Audio file transcription | ✅ | WAV file transcribe |
| Threaded processing | ✅ | Runs on separate thread |
| Voice Activity Detection (VAD) | ⬜ Planned | Silero VAD built into whisper.cpp |
| Flash Attention | ✅ | Enabled by default since v1.8.0 |
| Model downloading in editor | ✅ | Download models from Godot editor |
| Quantized models | ✅ | Q5_0, Q5_1, Q8_0 support |

## How to install

Go to a github release, copy paste the addons folder to the samples folder. Restart godot editor.

</p>
<p align="center">
<img src="banner_godot_whisper.jpg"/>
</p>

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

## Video Tutorial

[![Comparison](https://img.youtube.com/vi/fAgjNkfBOKs/0.jpg)](https://www.youtube.com/watch?v=fAgjNkfBOKs&t=10s)

## How to build

```
scons target=template_release generate_bindings=no arch=universal precision=single
rm -rf samples/godot_whisper/addons
cp -rf bin/addons samples/godot_whisper/addons
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Ughuuu"><img src="https://avatars.githubusercontent.com/u/2369380?v=4?s=100" width="100px;" alt="Dragos Daian"/><br /><sub><b>Dragos Daian</b></sub></a><br /><a href="https://github.com/appsinacup/appsinacup.whisper/commits?author=Ughuuu" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://chibifire.com"><img src="https://avatars.githubusercontent.com/u/32321?v=4?s=100" width="100px;" alt="K. S. Ernest (iFire) Lee"/><br /><sub><b>K. S. Ernest (iFire) Lee</b></sub></a><br /><a href="https://github.com/appsinacup/appsinacup.whisper/commits?author=fire" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
