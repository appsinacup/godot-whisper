# Copilot Instructions — Godot Whisper

## Project Overview
Godot Whisper is a GDExtension plugin that integrates [whisper.cpp](https://github.com/ggml-org/whisper.cpp) into the Godot Engine for real-time and offline speech-to-text transcription. It targets Godot 4.1+ via godot-cpp.

## Architecture
- **Build system**: SCons (`SConstruct` at root). No CMake for the plugin itself.
- **Language**: C++ (godot-cpp style). Source files live in `src/`.
- **Third-party deps** are git submodules under `thirdparty/`:
  - `whisper.cpp` (v1.8.4, ggml-org/whisper.cpp) — the ML inference engine
  - `godot-cpp` (branch 4.1) — Godot C++ bindings
  - `opencl_headers`, `opencl_icd_loader` — OpenCL support (headers & ICD loader)
  - `libsamplerate` (0.1.9) — audio resampling
- After cloning, run `git submodule update --init --recursive`.

## Key Source Files
| File | Purpose |
|------|---------|
| `src/speech_to_text.h/.cpp` | Main `SpeechToText` node — transcription, VAD, resampling |
| `src/resource_whisper.h/.cpp` | `ResourceWhisper` — wraps a whisper model binary |
| `src/resource_loader_whisper.h/.cpp` | Custom ResourceLoader for `.bin` model files |
| `src/register_types.h/.cpp` | GDExtension entry point — class registration |
| `SConstruct` | Build script (all platforms) |
| `include/config.h`, `include/build_config.h` | Build-time configuration |

## whisper.cpp v1.8.4 Layout
The submodule has a new directory structure (changed from the old monolithic layout):
- `include/whisper.h` — public API header
- `src/whisper.cpp` — whisper implementation
- `ggml/include/` — ggml public headers (ggml.h, ggml-cpu.h, ggml-metal.h, ggml-backend.h, …)
- `ggml/src/` — ggml core (ggml.c, ggml.cpp, ggml-backend.cpp, ggml-alloc.c, …)
- `ggml/src/ggml-cpu/` — CPU backend (+ arch/ subdirs for arm, x86, wasm, …)
- `ggml/src/ggml-metal/` — Metal backend (multiple .cpp/.m files)
- `ggml/src/ggml-opencl/` — OpenCL backend (self-contained, embedded kernels)
- `ggml/src/ggml-vulkan/` — Vulkan backend (auto-detected when `glslc` available)
- `ggml/src/ggml-webgpu/` — WebGPU backend (opt-in, requires Emscripten 4.0+)

## Platform / Backend Matrix
| Platform | Backend | SConstruct branch | SCons flag |
|----------|---------|-------------------|------------|
| macOS / iOS | Metal + Accelerate | `env["platform"] in ["macos", "ios"]` | (always) |
| Linux / Windows / Android | OpenCL (self-contained) | `else` branch | (always) |
| Linux / Windows / Android | Vulkan (SPIR-V shaders) | `else` branch | auto-detected (default) |
| Web | CPU-only (WASM) | `env["platform"] == "web"` | (default) |
| Web | WebGPU (WGSL shaders) | `env["platform"] == "web"` | `webgpu=yes` |

## Coding Conventions
- Follow godot-cpp naming: `snake_case` for methods, `PascalCase` for classes.
- Private helpers prefixed with `_` (e.g. `_is_use_gpu()`, `_get_language()`).
- Use `ERR_PRINT` / `WARN_PRINT` / `UtilityFunctions::print` for logging, never raw `printf`/`std::cout`.
- Godot types (`PackedFloat32Array`, `String`, `Array`, `Dictionary`) over STL equivalents at the API boundary.
- Includes: use `<whisper.cpp/include/whisper.h>` and `<libsamplerate/src/samplerate.h>` (resolved via `thirdparty` in CPPPATH).

## Build Commands
```bash
# macOS (Apple Silicon, release)
scons target=template_release arch=arm64 precision=single

# macOS (universal)
scons target=template_release arch=universal precision=single

# Linux (OpenCL + Vulkan auto-detected)
scons target=template_release arch=x86_64 precision=single

# Windows (from MSVC shell)
scons target=template_release arch=x86_64 precision=single

# Web (WebGPU — requires Emscripten 4.0+)
scons target=template_release arch=wasm32 precision=single webgpu=yes

# Web (CPU-only, default)
scons target=template_release arch=wasm32 precision=single

# Copy to sample project
cp -rf bin/addons samples/godot_whisper/addons
```

## Important API Notes (whisper.cpp v1.8.4)
- `whisper_context_params` — use `whisper_context_default_params()`, then set fields. Do NOT brace-init.
- `speed_up` field was **removed** from `whisper_full_params`.
- `suppress_non_speech_tokens` was **renamed** to `suppress_nst`.
- `flash_attn` is now a field on `whisper_context_params` (enabled by default).
- VAD API: `whisper_vad_default_context_params()` (new in v1.8.x).

## Git Workflow
- Do NOT make git commits automatically. Only commit when the user explicitly asks.
- Branch: `update-to-latest-whisper-cpp` is the active development branch.
- All thirdparty dependencies are git submodules — never vendor/copy files manually.

## CI
GitHub Actions workflows in `.github/workflows/` build for all platforms (macOS, iOS, Linux, Windows, Android, Web) with both single and double precision.

# **Always end interactions with confirmation.**
After completing any work or providing information, use `ask_questions` to confirm the task is complete and ask if anything else is needed. Never finish a turn without this confirmation.
