R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
		R"(
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(// cpy
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(
)"
		R"(kernel void kernel_cpy_f16_f16(
)"
		R"(        global half * src0,
)"
		R"(        ulong offset0,
)"
		R"(        global half * dst,
)"
		R"(        ulong offsetd,
)"
		R"(        int ne00,
)"
		R"(        int ne01,
)"
		R"(        int ne02,
)"
		R"(        int ne03,
)"
		R"(        ulong nb00,
)"
		R"(        ulong nb01,
)"
		R"(        ulong nb02,
)"
		R"(        ulong nb03,
)"
		R"(        int ne0,
)"
		R"(        int ne1,
)"
		R"(        int ne2,
)"
		R"(        int ne3,
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
		R"(    src0 = (global half*)((global char*)src0 + offset0);
)"
		R"(    dst = (global half*)((global char*)dst + offsetd);
)"
		R"(
)"
		R"(    int i03 = get_group_id(2);
)"
		R"(    int i02 = get_group_id(1);
)"
		R"(    int i01 = get_group_id(0);
)"
		R"(
)"
		R"(    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
)"
		R"(
)"
		R"(    int i3 = n / (ne2*ne1*ne0);
)"
		R"(    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
)"
		R"(    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
)"
		R"(    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);
)"
		R"(
)"
		R"(    global half * dst_data = (global half *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
)"
		R"(
)"
		R"(    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
)"
		R"(        global const half * src = (global half *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
)"
		R"(        dst_data[i00] = src[0];
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_cpy_f16_f32(
)"
		R"(        global half * src0,
)"
		R"(        ulong offset0,
)"
		R"(        global float * dst,
)"
		R"(        ulong offsetd,
)"
		R"(        int ne00,
)"
		R"(        int ne01,
)"
		R"(        int ne02,
)"
		R"(        int ne03,
)"
		R"(        ulong nb00,
)"
		R"(        ulong nb01,
)"
		R"(        ulong nb02,
)"
		R"(        ulong nb03,
)"
		R"(        int ne0,
)"
		R"(        int ne1,
)"
		R"(        int ne2,
)"
		R"(        int ne3,
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
		R"(
)"
		R"(    src0 = (global half*)((global char*)src0 + offset0);
)"
		R"(    dst = (global float*)((global char*)dst + offsetd);
)"
		R"(
)"
		R"(    int i03 = get_group_id(2);
)"
		R"(    int i02 = get_group_id(1);
)"
		R"(    int i01 = get_group_id(0);
)"
		R"(
)"
		R"(    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
)"
		R"(
)"
		R"(    int i3 = n / (ne2*ne1*ne0);
)"
		R"(    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
)"
		R"(    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
)"
		R"(    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);
)"
		R"(
)"
		R"(    global float * dst_data = (global float *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
)"
		R"(
)"
		R"(    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
)"
		R"(        global half * src = (global half *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
)"
		R"(        dst_data[i00] = src[0];
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_cpy_f32_f16(
)"
		R"(        global float * src0,
)"
		R"(        ulong offset0,
)"
		R"(        global half * dst,
)"
		R"(        ulong offsetd,
)"
		R"(        int ne00,
)"
		R"(        int ne01,
)"
		R"(        int ne02,
)"
		R"(        int ne03,
)"
		R"(        ulong nb00,
)"
		R"(        ulong nb01,
)"
		R"(        ulong nb02,
)"
		R"(        ulong nb03,
)"
		R"(        int ne0,
)"
		R"(        int ne1,
)"
		R"(        int ne2,
)"
		R"(        int ne3,
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
		R"(    src0 = (global float*)((global char*)src0 + offset0);
)"
		R"(    dst = (global half*)((global char*)dst + offsetd);
)"
		R"(
)"
		R"(    int i03 = get_group_id(2);
)"
		R"(    int i02 = get_group_id(1);
)"
		R"(    int i01 = get_group_id(0);
)"
		R"(
)"
		R"(    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
)"
		R"(
)"
		R"(    int i3 = n / (ne2*ne1*ne0);
)"
		R"(    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
)"
		R"(    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
)"
		R"(    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);
)"
		R"(
)"
		R"(    global half * dst_data = (global half *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
)"
		R"(
)"
		R"(    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
)"
		R"(        global const float * src = (global float *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
)"
		R"(
)"
		R"(        dst_data[i00] = src[0];
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_cpy_f32_f32(
)"
		R"(        global float * src0,
)"
		R"(        ulong offset0,
)"
		R"(        global float * dst,
)"
		R"(        ulong offsetd,
)"
		R"(        int ne00,
)"
		R"(        int ne01,
)"
		R"(        int ne02,
)"
		R"(        int ne03,
)"
		R"(        ulong nb00,
)"
		R"(        ulong nb01,
)"
		R"(        ulong nb02,
)"
		R"(        ulong nb03,
)"
		R"(        int ne0,
)"
		R"(        int ne1,
)"
		R"(        int ne2,
)"
		R"(        int ne3,
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
		R"(    src0 = (global float*)((global char*)src0 + offset0);
)"
		R"(    dst = (global float*)((global char*)dst + offsetd);
)"
		R"(
)"
		R"(    int i03 = get_group_id(2);
)"
		R"(    int i02 = get_group_id(1);
)"
		R"(    int i01 = get_group_id(0);
)"
		R"(
)"
		R"(    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
)"
		R"(
)"
		R"(    int i3 = n / (ne2*ne1*ne0);
)"
		R"(    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
)"
		R"(    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
)"
		R"(    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);
)"
		R"(
)"
		R"(    global float * dst_data = (global float *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
)"
		R"(
)"
		R"(    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
)"
		R"(        global const float * src = (global float *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
)"
		R"(
)"
		R"(        dst_data[i00] = src[0];
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_cpy_i32_i32(
)"
		R"(        global int * src0,
)"
		R"(        ulong offset0,
)"
		R"(        global int * dst,
)"
		R"(        ulong offsetd,
)"
		R"(        int ne00,
)"
		R"(        int ne01,
)"
		R"(        int ne02,
)"
		R"(        int ne03,
)"
		R"(        ulong nb00,
)"
		R"(        ulong nb01,
)"
		R"(        ulong nb02,
)"
		R"(        ulong nb03,
)"
		R"(        int ne0,
)"
		R"(        int ne1,
)"
		R"(        int ne2,
)"
		R"(        int ne3,
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
		R"(    src0 = (global int*)((global char*)src0 + offset0);
)"
		R"(    dst = (global int*)((global char*)dst + offsetd);
)"
		R"(
)"
		R"(    int i03 = get_group_id(2);
)"
		R"(    int i02 = get_group_id(1);
)"
		R"(    int i01 = get_group_id(0);
)"
		R"(
)"
		R"(    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
)"
		R"(
)"
		R"(    int i3 = n / (ne2*ne1*ne0);
)"
		R"(    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
)"
		R"(    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
)"
		R"(    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);
)"
		R"(
)"
		R"(    global int * dst_data = (global int *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
)"
		R"(
)"
		R"(    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
)"
		R"(        global const int * src = (global int *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
)"
		R"(
)"
		R"(        dst_data[i00] = src[0];
)"
		R"(    }
)"
		R"(}
)"
