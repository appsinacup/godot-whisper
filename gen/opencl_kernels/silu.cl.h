R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// silu
)"
R"(//------------------------------------------------------------------------------
)"
R"(kernel void kernel_silu(
)"
R"(        global float * src0,
)"
R"(        ulong offset0,
)"
R"(        global float * dst,
)"
R"(        ulong offsetd
)"
R"() {
)"
R"(    src0 = (global float*)((global char*)src0 + offset0);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    float x = src0[get_global_id(0)];
)"
R"(    dst[get_global_id(0)] = x / (1.0f + exp(-x));
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_silu_4(
)"
R"(        global float4 * src0,
)"
R"(        ulong offset0,
)"
R"(        global float4 * dst,
)"
R"(        ulong offsetd
)"
R"() {
)"
R"(    src0 = (global float4*)((global char*)src0 + offset0);
)"
R"(    dst = (global float4*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    float4 x = src0[get_global_id(0)];
)"
R"(    dst[get_global_id(0)] = x / (1.0f + exp(-x));
)"
R"(}
)"
