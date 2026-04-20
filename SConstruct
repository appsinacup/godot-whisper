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
        "GGML_USE_CPU",
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
    ggml_src + "/ggml-backend-dl.cpp",
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
    if env["arch"] == "universal":
        # Universal builds use "-arch x86_64 -arch arm64" so every source is
        # compiled for BOTH architectures.  Arch-specific files (e.g.
        # arch/arm/quants.c) have per-function #ifdef __ARM_NEON guards with
        # scalar #else fallbacks, so the x86_64 slice produces non-generic
        # symbol names that collide with arch-fallback.h renames in the base
        # quants.c / repack.cpp.  Fix: compile each arch dir with only its
        # target -arch flag so the other slice is never emitted.
        def _single_arch_env(base_env, keep_arch):
            """Return a clone of base_env with only one -arch flag."""
            e = base_env.Clone()
            for key in ("CCFLAGS", "LINKFLAGS"):
                old = list(e.get(key, []))
                new = []
                i = 0
                while i < len(old):
                    if str(old[i]) == "-arch" and i + 1 < len(old):
                        if str(old[i + 1]) == keep_arch:
                            new.extend(["-arch", keep_arch])
                        i += 2
                    else:
                        new.append(old[i])
                        i += 1
                e[key] = new
            return e

        arm_env = _single_arch_env(env, "arm64")
        x86_env = _single_arch_env(env, "x86_64")

        cpu_sources.extend([
            arm_env.SharedObject(cpu_dir + "/arch/arm/quants.c"),
            arm_env.SharedObject(cpu_dir + "/arch/arm/repack.cpp"),
            arm_env.SharedObject(cpu_dir + "/arch/arm/cpu-feats.cpp"),
            x86_env.SharedObject(cpu_dir + "/arch/x86/quants.c"),
            x86_env.SharedObject(cpu_dir + "/arch/x86/repack.cpp"),
            x86_env.SharedObject(cpu_dir + "/arch/x86/cpu-feats.cpp"),
        ])
    elif env["arch"] == "x86_64":
        cpu_sources.append(cpu_dir + "/arch/x86/quants.c")
        cpu_sources.append(cpu_dir + "/arch/x86/repack.cpp")
        cpu_sources.append(cpu_dir + "/arch/x86/cpu-feats.cpp")
    else:
        # arm64
        cpu_sources.append(cpu_dir + "/arch/arm/quants.c")
        cpu_sources.append(cpu_dir + "/arch/arm/repack.cpp")
        cpu_sources.append(cpu_dir + "/arch/arm/cpu-feats.cpp")
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

# ── Patch upstream bugs in whisper.cpp submodule ──────────────────────────────
# sgemm.cpp has a known fp16 NEON guard bug (FIXME in source) that breaks arm32.
# It checks !defined(_MSC_VER) instead of __ARM_FEATURE_FP16_VECTOR_ARITHMETIC.
# Patch at build time so CI works without a custom submodule fork.
_sgemm_path = cpu_dir + "/llamafile/sgemm.cpp"
with open(_sgemm_path, "r") as f:
    _sgemm_txt = f.read()
_sgemm_orig = '#if !defined(_MSC_VER)\n// FIXME: this should check for __ARM_FEATURE_FP16_VECTOR_ARITHMETIC'
if _sgemm_orig in _sgemm_txt:
    _sgemm_txt = _sgemm_txt.replace(
        _sgemm_orig,
        '#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)')
    _sgemm_txt = _sgemm_txt.replace(
        '#endif // _MSC_VER\n#endif // __ARM_NEON',
        '#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC\n#endif // __ARM_NEON')
    with open(_sgemm_path, "w") as f:
        f.write(_sgemm_txt)

# ── whisper.cpp library itself ────────────────────────────────────────────────
sources.append(whisper_dir + "/src/whisper.cpp")

# ── Disable narrowing warning (Clang) ─────────────────────────────────────────
# whisper.cpp/ggml has narrowing conversions that break on 32-bit (size_t = u32)
if not env.get("is_msvc", False):
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
            "GGML_METAL_EMBED_LIBRARY",
            "GGML_USE_ACCELERATE",
            "ACCELERATE_NEW_LAPACK",
            "ACCELERATE_LAPACK_ILP64",
        ]
    )
    env.Append(CPPPATH=[metal_dir])

    # ── Embed Metal shader library into the binary ────────────────────────────
    # Replicates the CMake GGML_METAL_EMBED_LIBRARY logic:
    #   1. Inline ggml-common.h and ggml-metal-impl.h into ggml-metal.metal
    #   2. Create an assembly file that .incbin's the result
    #   3. Compile & link the assembly so the shader source is available at runtime
    _metal_gen_dir = "gen/metal"
    os.makedirs(_metal_gen_dir, exist_ok=True)

    _metal_source      = metal_dir + "/ggml-metal.metal"
    _metal_impl_h      = metal_dir + "/ggml-metal-impl.h"
    _ggml_common_h     = ggml_src + "/ggml-common.h"
    _metal_embed_metal = os.path.abspath(os.path.join(_metal_gen_dir, "ggml-metal-embed.metal"))
    _metal_embed_asm   = os.path.join(_metal_gen_dir, "ggml-metal-embed.s")

    # Check if regeneration is needed
    _metal_deps = [_metal_source, _metal_impl_h, _ggml_common_h]
    _needs_metal_regen = (
        not os.path.exists(_metal_embed_asm) or
        any(os.path.getmtime(d) > os.path.getmtime(_metal_embed_asm) for d in _metal_deps)
    )
    if _needs_metal_regen:
        # Read sources
        with open(_metal_source, "r") as f:
            _mtl = f.read()
        with open(_ggml_common_h, "r") as f:
            _common_h = f.read()
        with open(_metal_impl_h, "r") as f:
            _impl_h = f.read()

        # Inline the includes (same substitutions as CMake sed commands)
        _mtl = _mtl.replace("__embed_ggml-common.h__", _common_h)
        _mtl = _mtl.replace('#include "ggml-metal-impl.h"', _impl_h)

        with open(_metal_embed_metal, "w") as f:
            f.write(_mtl)

        # Generate assembly that embeds the processed .metal source
        with open(_metal_embed_asm, "w") as f:
            f.write('.section __DATA,__ggml_metallib\n')
            f.write('.globl _ggml_metallib_start\n')
            f.write('_ggml_metallib_start:\n')
            f.write('.incbin "' + _metal_embed_metal + '"\n')
            f.write('.globl _ggml_metallib_end\n')
            f.write('_ggml_metallib_end:\n')

    # Compile the metal embed assembly using the C compiler (clang) rather than
    # the raw 'as' assembler.  'as' on macOS defaults to a macOS target and does
    # not receive the iOS SDK / -target flags that clang carries, which causes
    # the linker to reject the object when building for iOS:
    #   "ld: building for 'iOS', but linking in object file built for 'macOS'"
    _metal_embed_obj = env.Command(
        "gen/metal/ggml-metal-embed.os",
        _metal_embed_asm,
        "$CC $CCFLAGS -c -o $TARGET $SOURCE",
    )

    # ggml-metal-device has both .cpp and .m — give the .m an explicit object name
    metal_sources = [
        metal_dir + "/ggml-metal.cpp",
        metal_dir + "/ggml-metal-common.cpp",
        metal_dir + "/ggml-metal-device.cpp",
        env.SharedObject(metal_dir + "/ggml-metal-device_m", metal_dir + "/ggml-metal-device.m"),
        metal_dir + "/ggml-metal-context.m",
        metal_dir + "/ggml-metal-ops.cpp",
        _metal_embed_obj,
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
            ("GGML_OPENCL_TARGET_VERSION", 300),
        ]
    )

    # ── OpenCL ICD Loader (statically linked from submodule) ─────────────
    # Instead of depending on system libOpenCL.so.1 / OpenCL.dll, we build
    # the Khronos ICD loader directly and link it into the shared library.
    # At runtime the ICD loader discovers vendor drivers via:
    #   Linux/Android: /etc/OpenCL/vendors/*.icd (or /system/vendor/...)
    #   Windows:       Registry HKLM\...\Khronos\OpenCL\Vendors
    icd_loader_dir = "thirdparty/opencl_icd_loader/loader"
    icd_gen_dir = "gen/opencl_icd"
    os.makedirs(icd_gen_dir, exist_ok=True)

    # Generate icd_cmake_config.h (CMake normally does this via configure_file)
    _icd_config_path = os.path.join(icd_gen_dir, "icd_cmake_config.h")
    _icd_config_lines = []
    # Linux glibc has secure_getenv when _GNU_SOURCE is defined (already set).
    # Android Bionic added it in API 28+; skip to avoid issues on older NDKs.
    if env["platform"] == "linux":
        _icd_config_lines.append("#define HAVE_SECURE_GETENV")
    _icd_config_content = "\n".join(_icd_config_lines) + "\n"
    # Only rewrite if content changed to avoid unnecessary rebuilds
    _existing = ""
    if os.path.exists(_icd_config_path):
        with open(_icd_config_path, "r") as f:
            _existing = f.read()
    if _existing != _icd_config_content:
        with open(_icd_config_path, "w") as f:
            f.write(_icd_config_content)

    # Common ICD loader sources
    icd_sources = [
        icd_loader_dir + "/icd.c",
        icd_loader_dir + "/icd_dispatch.c",
        icd_loader_dir + "/icd_dispatch_generated.c",
        icd_loader_dir + "/icd_trace.c",
    ]

    # Platform-specific ICD loader sources
    if env["platform"] == "windows":
        icd_sources.extend([
            icd_loader_dir + "/windows/icd_windows.c",
            icd_loader_dir + "/windows/icd_windows_dxgk.c",
            icd_loader_dir + "/windows/icd_windows_library.c",
            icd_loader_dir + "/windows/icd_windows_envvars.c",
            icd_loader_dir + "/windows/icd_windows_hkr.c",
            icd_loader_dir + "/windows/icd_windows_apppackage.c",
        ])
    else:
        # Linux and Android (both use the linux/ ICD platform code)
        icd_sources.extend([
            icd_loader_dir + "/linux/icd_linux.c",
            icd_loader_dir + "/linux/icd_linux_library.c",
            icd_loader_dir + "/linux/icd_linux_envvars.c",
        ])

    # Build ICD loader with isolated defines (PRIVATE to the loader, not ggml-opencl)
    icd_env = env.Clone()
    icd_env.Append(CPPPATH=[icd_loader_dir, icd_gen_dir, "thirdparty/opencl_icd_loader/include"])
    icd_env.Append(CPPDEFINES=[
        ("CL_TARGET_OPENCL_VERSION", 300),
        "CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES",
        ("OPENCL_ICD_LOADER_VERSION_MAJOR", 3),
        ("OPENCL_ICD_LOADER_VERSION_MINOR", 0),
        ("OPENCL_ICD_LOADER_VERSION_REV", 8),
    ])
    sources.extend([icd_env.SharedObject(s) for s in icd_sources])

    # Platform link libraries needed by the ICD loader and ggml-cpu
    if env["platform"] == "windows":
        # cfgmgr32/runtimeobject: ICD loader vendor enumeration
        # advapi32: ggml-cpu.cpp uses Windows Registry APIs for CPU feature detection
        # ole32: ICD loader HKR vendor enumeration (StringFromGUID2)
        env.Append(LIBS=["cfgmgr32", "runtimeobject", "advapi32", "ole32"])
    else:
        # dl for dlopen (loading vendor .so)
        env.Append(LIBS=["dl"])
        # pthread for thread safety (Android Bionic has it built-in, no -lpthread needed)
        if env["platform"] != "android":
            env.Append(LIBS=["pthread"])

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

        # On Windows, CMD needs ".\gen\..." to run local executables.
        # On Unix, "./" prefix is needed.  Use os.sep for the path separator.
        if sys.platform == "win32":
            _tool_cmd = vulkan_gen_tool_bin.replace("/", "\\\\")
        else:
            _tool_cmd = "./" + vulkan_gen_tool_bin

        # Generate header with extern declarations (no glslc needed, fast)
        vulkan_hdr = env.Command(
            vulkan_header_path,
            vulkan_tool,
            "{tool} --output-dir {spvdir} --target-hpp $TARGET".format(
                tool=_tool_cmd, spvdir=vulkan_spv_dir)
        )

        # Compile each .comp shader → .cpp with embedded SPIR-V
        import glob as _vk_glob
        for _comp_path in sorted(_vk_glob.glob(vulkan_shader_src + "/*.comp")):
            _comp_name = os.path.basename(_comp_path)
            _cpp_out = os.path.join(vulkan_gen_dir, _comp_name + ".cpp")
            _shader_cmd = env.Command(
                _cpp_out,
                _comp_path,
                "{tool} --glslc {glslc} --source $SOURCE "
                "--output-dir {spvdir} --target-hpp {hdr} --target-cpp $TARGET".format(
                    tool=_tool_cmd, glslc=_glslc_path,
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
