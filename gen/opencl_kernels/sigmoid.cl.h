R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// sigmoid
)"
R"(//------------------------------------------------------------------------------
)"
R"(
)"
R"(kernel void kernel_sigmoid_f32(
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
R"(    dst[get_global_id(0)] = 1.0f / (1.0f + exp(-src0[get_global_id(0)]));
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_sigmoid_f16(
)"
R"(        global half * src0,
)"
R"(        ulong offset0,
)"
R"(        global half * dst,
)"
R"(        ulong offsetd
)"
R"() {
)"
R"(    src0 = (global half*)((global char*)src0 + offset0);
)"
R"(    dst = (global half*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    dst[get_global_id(0)] = 1.0f / (1.0f + exp(-src0[get_global_id(0)]));
)"
R"(}
)"
