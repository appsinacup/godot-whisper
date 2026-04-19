R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// relu
)"
R"(//------------------------------------------------------------------------------
)"
R"(kernel void kernel_relu(
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
R"(    dst[get_global_id(0)] = fmax(0.0f, src0[get_global_id(0)]);
)"
R"(}
)"
