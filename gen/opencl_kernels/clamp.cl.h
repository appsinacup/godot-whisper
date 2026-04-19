R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// clamp
)"
R"(//------------------------------------------------------------------------------
)"
R"(kernel void kernel_clamp(
)"
R"(        global float * src0,
)"
R"(        ulong offset0,
)"
R"(        global float * dst,
)"
R"(        ulong offsetd,
)"
R"(        float min,
)"
R"(        float max
)"
R"() {
)"
R"(    src0 = (global float*)((global char*)src0 + offset0);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    dst[get_global_id(0)] = src0[get_global_id(0)] < min ?
)"
R"(        min :
)"
R"(        (src0[get_global_id(0)] > max ? max : src0[get_global_id(0)]);
)"
R"(}
)"
