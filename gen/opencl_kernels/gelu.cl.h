R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
		R"(
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(// gelu
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(#define GELU_COEF_A     0.044715f
)"
		R"(#define GELU_QUICK_COEF -1.702f
)"
		R"(#define SQRT_2_OVER_PI  0.79788456080286535587989211986876f
)"
		R"(#define SQRT_2_INV      0.70710678118654752440084436210484f
)"
		R"(
)"
		R"(kernel void kernel_gelu(
)"
		R"(    global float * src0,
)"
		R"(    ulong offset0,
)"
		R"(    global float * dst,
)"
		R"(    ulong offsetd
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
		R"(
)"
		R"(    dst[get_global_id(0)] = 0.5f*x*(1.0f + tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_gelu_4(
)"
		R"(    global float4 * src0,
)"
		R"(    ulong offset0,
)"
		R"(    global float4 * dst,
)"
		R"(    ulong offsetd
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
		R"(
)"
		R"(    dst[get_global_id(0)] = 0.5f*x*(1.0f + tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_gelu_erf(
)"
		R"(    global float * src0,
)"
		R"(    ulong offset0,
)"
		R"(    global float * dst,
)"
		R"(    ulong offsetd
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
		R"(    dst[get_global_id(0)] = 0.5f*x*(1.0f + erf(x*SQRT_2_INV));
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_gelu_erf_4(
)"
		R"(    global float4 * src0,
)"
		R"(    ulong offset0,
)"
		R"(    global float4 * dst,
)"
		R"(    ulong offsetd
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
		R"(    dst[get_global_id(0)] = 0.5f*x*(1.0f + erf(x*SQRT_2_INV));
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_gelu_quick(
)"
		R"(    global float * src0,
)"
		R"(    ulong offset0,
)"
		R"(    global float * dst,
)"
		R"(    ulong offsetd
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
		R"(    dst[get_global_id(0)] = x*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x)));
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_gelu_quick_4(
)"
		R"(    global float4 * src0,
)"
		R"(    ulong offset0,
)"
		R"(    global float4 * dst,
)"
		R"(    ulong offsetd
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
		R"(    dst[get_global_id(0)] = x*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x)));
)"
		R"(}
)"
