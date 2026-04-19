R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// fill
)"
R"(//------------------------------------------------------------------------------
)"
R"(__kernel void kernel_fill_f32(
)"
R"(        __global float *dst,
)"
R"(        ulong offsetd,
)"
R"(        float v,
)"
R"(        int n
)"
R"(
)"
R"() {
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
)"
R"(    if(get_global_id(0) < n){
)"
R"(        dst[get_global_id(0)] = v;
)"
R"(    }
)"
R"(}
)"
