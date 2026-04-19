#!/usr/bin/env python
import os
import sys

env = SConscript("thirdparty/godot-cpp/SConstruct")
# Clone the env so our modifications don't leak to godot-cpp's pending builds.
env = env.Clone()

# whisper.cpp/ggml uses try/catch (e.g. ggml-backend-reg.cpp), so re-enable
# exceptions that godot-cpp 4.2+ disables by default.
if env.get("is_msvc", False):
    if ("_HAS_EXCEPTIONS", 0) in env.get("CPPDEFINES", []):
        env["CPPDEFINES"].remove(("_HAS_EXCEPTIONS", 0))
    env.Append(CXXFLAGS=["/EHsc"])
elif env["platform"] == "web":
    # Emscripten: godot-cpp uses -sSUPPORT_LONGJMP='wasm' (Wasm-based SJLJ),
    # which is incompatible with -fexceptions (Emscripten JS-based EH).
    # Use -fwasm-exceptions (native Wasm EH) which IS compatible with Wasm SJLJ.
    cxxflags = env.get("CXXFLAGS", [])
    if "-fno-exceptions" in cxxflags:
        cxxflags.remove("-fno-exceptions")
    env.Append(CXXFLAGS=["-fwasm-exceptions"])
    env.Append(LINKFLAGS=["-fwasm-exceptions"])
else:
    cxxflags = env.get("CXXFLAGS", [])
    if "-fno-exceptions" in cxxflags:
        cxxflags.remove("-fno-exceptions")
    env.Append(CXXFLAGS=["-fexceptions"])

# ggml-backend-reg.cpp uses std::filesystem which requires iOS 13.0+.
# godot-cpp defaults to ios_min_version=12.0, so bump it.
if env["platform"] == "ios":
    ccflags = env.get("CCFLAGS", [])
    for i, flag in enumerate(ccflags):
        if isinstance(flag, str) and "-miphoneos-version-min=" in flag:
            ccflags[i] = "-miphoneos-version-min=13.0"
            break
        if isinstance(flag, str) and "-mios-simulator-version-min=" in flag:
            ccflags[i] = "-mios-simulator-version-min=13.0"
            break

# ── Paths ─────────────────────────────────────────────────────────────────────
whisper_dir = "thirdparty/whisper.cpp"
ggml_dir    = whisper_dir + "/ggml"
ggml_src    = ggml_dir + "/src"
cpu_dir     = ggml_src + "/ggml-cpu"
metal_dir   = ggml_src + "/ggml-metal"
blas_dir    = ggml_src + "/ggml-blas"
opencl_dir  = ggml_src + "/ggml-opencl"

# ── Global defines ────────────────────────────────────────────────────────────
env.Append(
    CPPDEFINES=[
        "HAVE_CONFIG_H",
        "PACKAGE=",
        "VERSION=",
        "CPU_CLIPS_POSITIVE=0",
        "CPU_CLIPS_NEGATIVE=0",
        "WHISPER_BUILD",
        "GGML_BUILD",
        "_GNU_SOURCE",
        "WHISPER_SHARED",
        "GGML_SHARED",
        'GGML_VERSION=\\"0.9.8\\"',
        'GGML_COMMIT=\\"v1.8.4\\"',
        'WHISPER_VERSION=\\"1.8.4\\"',
    ]
)

# ── Include paths ─────────────────────────────────────────────────────────────
# "thirdparty" is first so #include <whisper.cpp/include/whisper.h> works,
# but we also add the precise dirs so whisper.h's #include "ggml.h" resolves.
env.Prepend(CPPPATH=[
    "thirdparty",
    "include",
    whisper_dir + "/include",      # whisper.h
    ggml_dir + "/include",         # ggml.h, ggml-cpu.h, ggml-backend.h, …
    ggml_src,                      # ggml-impl.h, ggml-common.h, ggml-backend-impl.h, …
    cpu_dir,                       # ggml-cpu-impl.h, common.h, etc.
])
env.Append(CPPPATH=["src/"])

# ── godot-whisper sources ─────────────────────────────────────────────────────
sources = [Glob("src/*.cpp")]

# ── libsamplerate ─────────────────────────────────────────────────────────────
sources.extend([Glob("thirdparty/libsamplerate/src/*.c")])

# ── ggml core (platform-independent) ─────────────────────────────────────────
# ggml.c and ggml.cpp share the same base name → SCons would produce the same
# object file.  Compile the .cpp variant with an explicit unique object name.
ggml_core_sources = [
    ggml_src + "/ggml.c",
    env.SharedObject(ggml_src + "/ggml_cpp", ggml_src + "/ggml.cpp"),
    ggml_src + "/ggml-alloc.c",
    ggml_src + "/ggml-backend.cpp",
    ggml_src + "/ggml-backend-reg.cpp",
    ggml_src + "/ggml-opt.cpp",
    ggml_src + "/ggml-quants.c",
    ggml_src + "/ggml-threading.cpp",
    ggml_src + "/gguf.cpp",
]
sources.extend(ggml_core_sources)

# ── ggml-cpu backend (always needed) ─────────────────────────────────────────
# Same base-name conflict: ggml-cpu.c / ggml-cpu.cpp
cpu_sources = [
    cpu_dir + "/ggml-cpu.c",
    env.SharedObject(cpu_dir + "/ggml-cpu_cpp", cpu_dir + "/ggml-cpu.cpp"),
    cpu_dir + "/repack.cpp",
    cpu_dir + "/hbm.cpp",
    cpu_dir + "/quants.c",
    cpu_dir + "/traits.cpp",
    cpu_dir + "/binary-ops.cpp",
    cpu_dir + "/unary-ops.cpp",
    cpu_dir + "/vec.cpp",
    cpu_dir + "/ops.cpp",
    cpu_dir + "/amx/amx.cpp",
    cpu_dir + "/amx/mmq.cpp",
    cpu_dir + "/llamafile/sgemm.cpp",
]

# ── Architecture-specific CPU files ──────────────────────────────────────────
if env["platform"] in ["macos", "ios"]:
    # Apple universal: include both ARM and x86 — guarded by #if defined(...)
    cpu_sources.append(cpu_dir + "/arch/arm/quants.c")
    cpu_sources.append(cpu_dir + "/arch/arm/repack.cpp")
    cpu_sources.append(cpu_dir + "/arch/arm/cpu-feats.cpp")
    cpu_sources.append(cpu_dir + "/arch/x86/quants.c")
    cpu_sources.append(cpu_dir + "/arch/x86/repack.cpp")
    cpu_sources.append(cpu_dir + "/arch/x86/cpu-feats.cpp")
elif env["platform"] == "android":
    if env["arch"] in ["arm64", "arm32"]:
        cpu_sources.append(cpu_dir + "/arch/arm/quants.c")
        cpu_sources.append(cpu_dir + "/arch/arm/repack.cpp")
        cpu_sources.append(cpu_dir + "/arch/arm/cpu-feats.cpp")
    else:
        cpu_sources.append(cpu_dir + "/arch/x86/quants.c")
        cpu_sources.append(cpu_dir + "/arch/x86/repack.cpp")
        cpu_sources.append(cpu_dir + "/arch/x86/cpu-feats.cpp")
elif env["platform"] == "web":
    cpu_sources.append(cpu_dir + "/arch/wasm/quants.c")
elif env["platform"] in ["linux", "windows"]:
    cpu_sources.append(cpu_dir + "/arch/x86/quants.c")
    cpu_sources.append(cpu_dir + "/arch/x86/repack.cpp")
    cpu_sources.append(cpu_dir + "/arch/x86/cpu-feats.cpp")

sources.extend(cpu_sources)

# ── whisper.cpp library itself ────────────────────────────────────────────────
sources.append(whisper_dir + "/src/whisper.cpp")

# ── Disable narrowing warning (Clang) ────────────────────────────────────────
if env["platform"] in ["macos", "ios"]:
    env.Append(CCFLAGS=["-Wno-c++11-narrowing"])

# ── Platform-specific backends ────────────────────────────────────────────────
if env["platform"] in ["macos", "ios"]:
    # ── Metal + Accelerate ────────────────────────────────────────────────────
    env.Append(LINKFLAGS=[
        "-framework", "Foundation",
        "-framework", "Metal",
        "-framework", "MetalKit",
        "-framework", "Accelerate",
    ])
    env.Append(
        CPPDEFINES=[
            "GGML_USE_METAL",
            "GGML_USE_ACCELERATE",
            "ACCELERATE_NEW_LAPACK",
            "ACCELERATE_LAPACK_ILP64",
            "GGML_METAL_PATH_RESOURCES=..",
        ]
    )
    env.Append(CPPPATH=[metal_dir])
    # ggml-metal-device has both .cpp and .m — give the .m an explicit object name
    metal_sources = [
        metal_dir + "/ggml-metal.cpp",
        metal_dir + "/ggml-metal-common.cpp",
        metal_dir + "/ggml-metal-device.cpp",
        env.SharedObject(metal_dir + "/ggml-metal-device_m", metal_dir + "/ggml-metal-device.m"),
        metal_dir + "/ggml-metal-context.m",
        metal_dir + "/ggml-metal-ops.cpp",
    ]
    sources.extend(metal_sources)

elif env["platform"] == "web":
    # ── Web: WebGPU (opt-in) or CPU-only ─────────────────────────────────
    # WebGPU requires emscripten 4.0.10+ with emdawnwebgpu port.
    # Godot's default emscripten is 3.1.62, so WebGPU is opt-in.
    # Enable with: scons webgpu=yes (or env WEBGPU=yes)
    _use_webgpu = ARGUMENTS.get("webgpu", os.environ.get("WEBGPU", "no")) in ["yes", "true", "1"]

    if _use_webgpu:
        webgpu_dir = ggml_src + "/ggml-webgpu"
        webgpu_shader_dir = webgpu_dir + "/wgsl-shaders"
        webgpu_gen_dir = "gen/webgpu_shaders"

        # Generate WGSL shader header using embed_wgsl.py
        os.makedirs(webgpu_gen_dir, exist_ok=True)
        embed_script = webgpu_shader_dir + "/embed_wgsl.py"
        shader_header = webgpu_gen_dir + "/ggml-wgsl-shaders.hpp"
        import glob as _wgsl_glob
        wgsl_files = _wgsl_glob.glob(webgpu_shader_dir + "/*.wgsl")
        needs_regen = not os.path.exists(shader_header) or any(
            os.path.getmtime(f) > os.path.getmtime(shader_header)
            for f in wgsl_files + [embed_script]
        )
        if needs_regen:
            import subprocess as _sp
            _sp.run([sys.executable, embed_script,
                     "--input_dir", webgpu_shader_dir,
                     "--output_file", shader_header], check=True)

        env.Append(CPPDEFINES=["GGML_USE_WEBGPU"])
        env.Append(CPPPATH=[webgpu_dir, webgpu_gen_dir])
        # Emscripten flags for Dawn WebGPU port
        # Use -fwasm-exceptions (not -fexceptions) for compatibility with godot-cpp's -sSUPPORT_LONGJMP='wasm'
        env.Append(CCFLAGS=["--use-port=emdawnwebgpu"])
        env.Append(LINKFLAGS=["--use-port=emdawnwebgpu", "-sASYNCIFY"])
        sources.append(webgpu_dir + "/ggml-webgpu.cpp")
    # else: CPU-only (no additional flags needed)

else:
    # ── Linux / Windows / Android: OpenCL (self-contained, no CLBlast) ───────

    # --- Embed OpenCL kernels at build time (avoids runtime .cl file loading) -
    import glob as _glob
    opencl_kernel_dir = opencl_dir + "/kernels"
    opencl_gen_dir = "gen/opencl_kernels"
    os.makedirs(opencl_gen_dir, exist_ok=True)
    for cl_path in sorted(_glob.glob(opencl_kernel_dir + "/*.cl")):
        cl_name = os.path.basename(cl_path)
        out_path = os.path.join(opencl_gen_dir, cl_name + ".h")
        # Regenerate only if source is newer than output
        if not os.path.exists(out_path) or os.path.getmtime(cl_path) > os.path.getmtime(out_path):
            with open(cl_path, "r") as f_in, open(out_path, "w") as f_out:
                for line in f_in:
                    f_out.write('R"({})"\n'.format(line))

    env.Prepend(CPPPATH=[
        "thirdparty/opencl_headers",
        opencl_dir,
        opencl_gen_dir,
    ])
    env.Append(
        CPPDEFINES=[
            "GGML_USE_OPENCL",
            "GGML_OPENCL_EMBED_KERNELS",
            "GGML_OPENCL_SOA_Q",
        ]
    )

    # OpenCL ICD library
    env.Append(LIBPATH=["OpenCL-SDK/install/lib"])

    opencl_include_dir = os.environ.get("OpenCL_INCLUDE_DIR")
    if opencl_include_dir:
        env.Append(CPPPATH=[opencl_include_dir])

    opencl_library = os.environ.get("OpenCL_LIBRARY")
    if opencl_library:
        env.Append(LIBS=[opencl_library])
    elif env["platform"] == "windows":
        env.Append(LIBS=[":OpenCL.dll"])
    elif env["platform"] == "linux":
        env.Append(LIBS=[":libOpenCL.so.1"])

    # ggml-opencl backend (self-contained in v1.8.4)
    sources.append(opencl_dir + "/ggml-opencl.cpp")

    # ── Vulkan (auto-enabled when glslc is available) ──────────────────────
    # Override with: scons vulkan=no (or env VULKAN=no) to disable
    # Note: Android NDK doesn't include vulkan/vulkan.hpp (C++ wrapper), so
    # Vulkan is skipped for Android; OpenCL is used instead.
    import shutil as _shutil
    vulkan_opt = ARGUMENTS.get("vulkan", os.environ.get("VULKAN", "auto"))

    if env["platform"] == "android":
        # Android NDK lacks vulkan.hpp C++ headers needed by ggml-vulkan.cpp
        if vulkan_opt in ["yes", "true", "1"]:
            print("WARNING: Vulkan not supported for Android (missing vulkan.hpp). Using OpenCL only.")
        _use_vulkan = False
    else:
        _glslc_path = None

        _vulkan_sdk = os.environ.get("VULKAN_SDK", "")
        if _vulkan_sdk:
            _candidate = os.path.join(_vulkan_sdk, "bin", "glslc")
            if sys.platform == "win32":
                _candidate += ".exe"
            if os.path.exists(_candidate):
                _glslc_path = _candidate
        if not _glslc_path:
            _glslc_path = _shutil.which("glslc")

        _use_vulkan = False
        if vulkan_opt in ["yes", "true", "1"]:
            _use_vulkan = True
            if not _glslc_path:
                print("ERROR: vulkan=yes but glslc not found. Install Vulkan SDK.")
                Exit(1)
        elif vulkan_opt == "auto":
            _use_vulkan = _glslc_path is not None

    if _use_vulkan:
        print("Vulkan backend enabled (glslc: {})".format(_glslc_path))
        vulkan_dir = ggml_src + "/ggml-vulkan"
        vulkan_shader_src = vulkan_dir + "/vulkan-shaders"
        vulkan_gen_dir = "gen/vulkan_shaders"
        vulkan_spv_dir = vulkan_gen_dir + "/spv"
        vulkan_gen_tool_src = vulkan_shader_src + "/vulkan-shaders-gen.cpp"
        vulkan_header_path = vulkan_gen_dir + "/ggml-vulkan-shaders.hpp"

        if sys.platform == "win32":
            vulkan_gen_tool_bin = "gen/vulkan-shaders-gen.exe"
        else:
            vulkan_gen_tool_bin = "gen/vulkan-shaders-gen"

        os.makedirs(vulkan_gen_dir, exist_ok=True)
        os.makedirs(vulkan_spv_dir, exist_ok=True)

        # Build the shader gen tool with host compiler
        _host_cxx = os.environ.get("HOST_CXX", "c++" if sys.platform != "win32" else "cl.exe")
        if sys.platform == "win32":
            _host_compile_cmd = "{} /std:c++17 /O2 /EHsc $SOURCE /Fe$TARGET".format(_host_cxx)
        else:
            _host_compile_cmd = "{} -std=c++17 -O2 -pthread $SOURCE -o $TARGET".format(_host_cxx)

        vulkan_tool = env.Command(
            vulkan_gen_tool_bin,
            vulkan_gen_tool_src,
            _host_compile_cmd
        )

        # Generate header with extern declarations (no glslc needed, fast)
        vulkan_hdr = env.Command(
            vulkan_header_path,
            vulkan_tool,
            "./{tool} --output-dir {spvdir} --target-hpp $TARGET".format(
                tool=vulkan_gen_tool_bin, spvdir=vulkan_spv_dir)
        )

        # Compile each .comp shader → .cpp with embedded SPIR-V
        import glob as _vk_glob
        for _comp_path in sorted(_vk_glob.glob(vulkan_shader_src + "/*.comp")):
            _comp_name = os.path.basename(_comp_path)
            _cpp_out = os.path.join(vulkan_gen_dir, _comp_name + ".cpp")
            _shader_cmd = env.Command(
                _cpp_out,
                _comp_path,
                "./{tool} --glslc {glslc} --source $SOURCE "
                "--output-dir {spvdir} --target-hpp {hdr} --target-cpp $TARGET".format(
                    tool=vulkan_gen_tool_bin, glslc=_glslc_path,
                    spvdir=vulkan_spv_dir, hdr=vulkan_header_path)
            )
            env.Depends(_shader_cmd, [vulkan_tool, vulkan_hdr])
            sources.append(_cpp_out)

        env.Append(CPPDEFINES=["GGML_USE_VULKAN"])
        env.Append(CPPPATH=[vulkan_dir, vulkan_gen_dir])

        if _vulkan_sdk:
            env.Append(CPPPATH=[os.path.join(_vulkan_sdk, "include")])
            env.Append(LIBPATH=[os.path.join(_vulkan_sdk, "lib")])

        if env["platform"] == "windows":
            env.Append(LIBS=["vulkan-1"])
        else:
            env.Append(LIBS=["vulkan"])

        sources.append(vulkan_dir + "/ggml-vulkan.cpp")

# ── Build shared library ─────────────────────────────────────────────────────
if env["platform"] in ["macos", "ios"]:
    library = env.SharedLibrary(
        "bin/addons/godot_whisper/bin/libgodot_whisper{}.framework/libgodot_whisper{}".format(
            env["suffix"], env["suffix"]
        ),
        source=sources,
    )
else:
    library = env.SharedLibrary(
        "bin/addons/godot_whisper/bin/libgodot_whisper{}{}".format(
            env["suffix"], env["SHLIBSUFFIX"]
        ),
        source=sources,
    )
Default(library)
