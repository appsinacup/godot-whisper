R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
		R"(
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(// add_id
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(kernel void kernel_add_id(
)"
		R"(    global char * src0,
)"
		R"(    ulong         offset0,
)"
		R"(    global char * src1,
)"
		R"(    ulong         offset1,
)"
		R"(    global char * src2,
)"
		R"(    ulong         offset2,
)"
		R"(    global char * dst,
)"
		R"(    ulong         offsetd,
)"
		R"(    ulong         nb01,
)"
		R"(    ulong         nb02,
)"
		R"(    ulong         nb11,
)"
		R"(    ulong         nb21,
)"
		R"(    int           ne0,
)"
		R"(    int           ne1
)"
		R"() {
)"
		R"(    src0 = (global char*)((global char*)src0 + offset0);
)"
		R"(    src1 = (global char*)((global char*)src1 + offset1);
)"
		R"(    src2 = (global char*)((global char*)src2 + offset2);
)"
		R"(    dst  = (global char*)((global char*)dst  + offsetd);
)"
		R"(
)"
		R"(    int i1 = get_group_id(0);
)"
		R"(    int i2 = get_group_id(1);
)"
		R"(
)"
		R"(    const int i11 = *((global const int *) (src2 + i1*sizeof(int) + i2*nb21));
)"
		R"(
)"
		R"(    const size_t nb1 = ne0 * sizeof(float);
)"
		R"(    const size_t nb2 = ne1 * nb1;
)"
		R"(
)"
		R"(    global float * dst_row  = (global float *)((global char *)dst  + i1*nb1 + i2*nb2);
)"
		R"(    global float * src0_row = (global float *)((global char *)src0 + i1*nb01 + i2*nb02);
)"
		R"(    global float * src1_row = (global float *)((global char *)src1 + i11*nb11);
)"
		R"(
)"
		R"(    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
)"
		R"(        dst_row[i0] = src0_row[i0] + src1_row[i0];
)"
		R"(    }
)"
		R"(}
)"
