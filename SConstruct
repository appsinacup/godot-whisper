#!/usr/bin/env python
import os
import sys

env = SConscript("thirdparty/godot-cpp/SConstruct")

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
    env.Object(ggml_src + "/ggml_cpp.os", ggml_src + "/ggml.cpp"),
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
    env.Object(cpu_dir + "/ggml-cpu_cpp.os", cpu_dir + "/ggml-cpu.cpp"),
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
    # Apple → ARM (Apple Silicon) or x86 (older Intel Macs)
    cpu_sources.append(cpu_dir + "/arch/arm/quants.c")
    cpu_sources.append(cpu_dir + "/arch/arm/repack.cpp")
    cpu_sources.append(cpu_dir + "/arch/arm/cpu-feats.cpp")
elif env["platform"] == "android":
    cpu_sources.append(cpu_dir + "/arch/arm/quants.c")
    cpu_sources.append(cpu_dir + "/arch/arm/repack.cpp")
    cpu_sources.append(cpu_dir + "/arch/arm/cpu-feats.cpp")
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
        env.Object(metal_dir + "/ggml-metal-device_m.os", metal_dir + "/ggml-metal-device.m"),
        metal_dir + "/ggml-metal-context.m",
        metal_dir + "/ggml-metal-ops.cpp",
    ]
    sources.extend(metal_sources)

elif env["platform"] == "web":
    # Web: CPU-only (no GPU backend available for now)
    pass

else:
    # ── Linux / Windows / Android: CLBlast + OpenCL ──────────────────────────
    # Enable C++ exceptions needed by CLBlast
    if env["platform"] == "windows":
        env.Append(CCFLAGS=["/EHsc"])
    else:
        env.Append(CCFLAGS=["-fexceptions"])

    env.Prepend(CPPPATH=[
        "thirdparty/opencl_headers",
        "thirdparty/clblast/include",
        "thirdparty/clblast/src",
        opencl_dir,
    ])
    env.Append(
        CPPDEFINES=[
            "GGML_USE_OPENCL",
            "OPENCL_API",
            "USE_ICD_LOADER",
        ]
    )

    # OpenCL ICD library
    env.Append(LIBPATH=["OpenCL-SDK/install/lib"])

    opencl_include_dir = os.environ.get("OpenCL_INCLUDE_DIR")
    if opencl_include_dir:
        env.Append(CPPDEFINES=[opencl_include_dir])

    opencl_library = os.environ.get("OpenCL_LIBRARY")
    if opencl_library:
        env.Append(LIBS=[opencl_library])
    elif env["platform"] == "windows":
        env.Append(LIBS=[":OpenCL.dll"])
    elif env["platform"] == "linux":
        env.Append(LIBS=[":libOpenCL.so.1"])

    # ggml-opencl backend (new v1.8.4 location)
    sources.append(opencl_dir + "/ggml-opencl.cpp")

    # CLBlast sources
    clblast_sources = [
        "thirdparty/clblast/src/database/database.cpp",
        "thirdparty/clblast/src/routines/common.cpp",
        "thirdparty/clblast/src/utilities/compile.cpp",
        "thirdparty/clblast/src/utilities/clblast_exceptions.cpp",
        "thirdparty/clblast/src/utilities/timing.cpp",
        "thirdparty/clblast/src/utilities/utilities.cpp",
        "thirdparty/clblast/src/api_common.cpp",
        "thirdparty/clblast/src/cache.cpp",
        "thirdparty/clblast/src/kernel_preprocessor.cpp",
        "thirdparty/clblast/src/routine.cpp",
        "thirdparty/clblast/src/tuning/configurations.cpp",
        "thirdparty/clblast/src/clblast.cpp",
        "thirdparty/clblast/src/clblast_c.cpp",
        "thirdparty/clblast/src/tuning/tuning_api.cpp",
    ]

    databases = [
        "copy", "pad", "padtranspose", "transpose", "xaxpy", "xdot",
        "xgemm", "xgemm_direct", "xgemv", "xgemv_fast", "xgemv_fast_rot",
        "xger", "invert", "gemm_routine", "trsv_routine", "xconvgemm",
    ]
    for db in databases:
        clblast_sources.append(
            "thirdparty/clblast/src/database/kernels/{0}/{0}.cpp".format(db)
        )

    sources.extend(clblast_sources)
    sources.extend(Glob("thirdparty/clblast/src/routines/level1/*.cpp"))
    sources.extend(Glob("thirdparty/clblast/src/routines/level2/*.cpp"))
    sources.extend(Glob("thirdparty/clblast/src/routines/level3/*.cpp"))
    sources.extend(Glob("thirdparty/clblast/src/routines/levelx/*.cpp"))
    sources.extend(Glob("thirdparty/clblast/src/tuners/*.cpp"))

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
