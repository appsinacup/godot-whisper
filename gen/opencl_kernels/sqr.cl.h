R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(
)"
R"(kernel void kernel_sqr_cont_f32(
)"
R"(    global float * src0,
)"
R"(    ulong          offset0,
)"
R"(    global float * dst,
)"
R"(    ulong          offsetd
)"
R"() {
)"
R"(    src0 = (global float*)((global char*)src0 + offset0);
)"
R"(    dst  = (global float*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    uint gid = get_global_id(0);
)"
R"(    dst[gid] = src0[gid] * src0[gid];
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_sqr_cont_f32_4(
)"
R"(    global float4 * src0,
)"
R"(    ulong           offset0,
)"
R"(    global float4 * dst,
)"
R"(    ulong           offsetd
)"
R"() {
)"
R"(    src0 = (global float4*)((global char*)src0 + offset0);
)"
R"(    dst  = (global float4*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    uint gid = get_global_id(0);
)"
R"(    dst[gid] = src0[gid] * src0[gid];
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_sqr_cont_f16(
)"
R"(    global half * src0,
)"
R"(    ulong         offset0,
)"
R"(    global half * dst,
)"
R"(    ulong         offsetd
)"
R"() {
)"
R"(    src0 = (global half*)((global char*)src0 + offset0);
)"
R"(    dst  = (global half*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    uint gid = get_global_id(0);
)"
R"(    dst[gid] = src0[gid] * src0[gid];
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_sqr_cont_f16_4(
)"
R"(    global half4 * src0,
)"
R"(    ulong          offset0,
)"
R"(    global half4 * dst,
)"
R"(    ulong          offsetd
)"
R"() {
)"
R"(    src0 = (global half4*)((global char*)src0 + offset0);
)"
R"(    dst  = (global half4*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    uint gid = get_global_id(0);
)"
R"(    dst[gid] = src0[gid] * src0[gid];
)"
R"(}
)"
