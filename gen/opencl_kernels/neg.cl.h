R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(
)"
R"(kernel void kernel_neg_f32(
)"
R"(        global const float * src0,
)"
R"(        ulong                offset0,
)"
R"(        global       float * dst,
)"
R"(        ulong                offsetd,
)"
R"(        int                  n
)"
R"() {
)"
R"(    if (get_global_id(0) >= n) {
)"
R"(        return;
)"
R"(    }
)"
R"(    src0 = (global float*)((global char*)src0 + offset0);
)"
R"(    dst  = (global float*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    dst[get_global_id(0)] = -src0[get_global_id(0)];
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_neg_f32_4(
)"
R"(        global const float4 * src0,
)"
R"(        ulong                 offset0,
)"
R"(        global       float4 * dst,
)"
R"(        ulong                 offsetd,
)"
R"(        int                   n
)"
R"() {
)"
R"(    if (get_global_id(0) >= n) {
)"
R"(        return;
)"
R"(    }
)"
R"(    src0 = (global float4*)((global char*)src0 + offset0);
)"
R"(    dst  = (global float4*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    dst[get_global_id(0)] = -src0[get_global_id(0)];
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_neg_f16(
)"
R"(        global const half * src0,
)"
R"(        ulong               offset0,
)"
R"(        global       half * dst,
)"
R"(        ulong               offsetd,
)"
R"(        int                 n
)"
R"() {
)"
R"(    if (get_global_id(0) >= n) {
)"
R"(        return;
)"
R"(    }
)"
R"(    src0 = (global half*)((global char*)src0 + offset0);
)"
R"(    dst  = (global half*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    dst[get_global_id(0)] = -src0[get_global_id(0)];
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_neg_f16_4(
)"
R"(        global const half4 * src0,
)"
R"(        ulong                offset0,
)"
R"(        global       half4 * dst,
)"
R"(        ulong                offsetd,
)"
R"(        int                  n
)"
R"() {
)"
R"(    if (get_global_id(0) >= n) {
)"
R"(        return;
)"
R"(    }
)"
R"(    src0 = (global half4*)((global char*)src0 + offset0);
)"
R"(    dst  = (global half4*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    dst[get_global_id(0)] = -src0[get_global_id(0)];
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_neg_f32_nc(
)"
R"(        global const char * src0,
)"
R"(        ulong               offset0,
)"
R"(        global       char * dst,
)"
R"(        ulong               offsetd,
)"
R"(        int   ne00,
)"
R"(        ulong nb00,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        ulong nb0,
)"
R"(        ulong nb1,
)"
R"(        ulong nb2,
)"
R"(        ulong nb3
)"
R"() {
)"
R"(    src0 = src0 + offset0;
)"
R"(    dst  = dst + offsetd;
)"
R"(
)"
R"(    const int i3 = get_group_id(2);
)"
R"(    const int i2 = get_group_id(1);
)"
R"(    const int i1 = get_group_id(0);
)"
R"(
)"
R"(    for (int i0 = get_local_id(0); i0 < ne00; i0 += get_local_size(0)) {
)"
R"(        global const float * x = (global const float *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
)"
R"(        global       float * y = (global       float *)(dst  + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
)"
R"(
)"
R"(        *y = -*x;
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_neg_f16_nc(
)"
R"(        global const char * src0,
)"
R"(        ulong               offset0,
)"
R"(        global       char * dst,
)"
R"(        ulong               offsetd,
)"
R"(        int   ne00,
)"
R"(        ulong nb00,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        ulong nb0,
)"
R"(        ulong nb1,
)"
R"(        ulong nb2,
)"
R"(        ulong nb3
)"
R"() {
)"
R"(    src0 = src0 + offset0;
)"
R"(    dst  = dst + offsetd;
)"
R"(
)"
R"(    const int i3 = get_group_id(2);
)"
R"(    const int i2 = get_group_id(1);
)"
R"(    const int i1 = get_group_id(0);
)"
R"(
)"
R"(    for (int i0 = get_local_id(0); i0 < ne00; i0 += get_local_size(0)) {
)"
R"(        global const half * x = (global const half *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
)"
R"(        global       half * y = (global       half *)(dst  + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
)"
R"(
)"
R"(        *y = -*x;
)"
R"(    }
)"
R"(}
)"
