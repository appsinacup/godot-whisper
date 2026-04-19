R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
		R"(
)"
		R"(kernel void kernel_scale_f32(
)"
		R"(        global float * src0,
)"
		R"(        ulong offset0,
)"
		R"(        global float * dst,
)"
		R"(        ulong offsetd,
)"
		R"(        float scale,
)"
		R"(        float bias
)"
		R"() {
)"
		R"(    src0 = (global float*)((global char*)src0 + offset0);
)"
		R"(    dst = (global float*)((global char*)dst + offsetd);
)"
		R"(    dst[get_global_id(0)] = src0[get_global_id(0)] * scale + bias;
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_scale_f32_4(
)"
		R"(        global float4 * src0,
)"
		R"(        ulong offset0,
)"
		R"(        global float4 * dst,
)"
		R"(        ulong offsetd,
)"
		R"(        float scale,
)"
		R"(        float bias
)"
		R"() {
)"
		R"(    src0 = (global float4*)((global char*)src0 + offset0);
)"
		R"(    dst = (global float4*)((global char*)dst + offsetd);
)"
		R"(    dst[get_global_id(0)] = src0[get_global_id(0)] * scale + bias;
)"
		R"(}
)"
