R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
		R"(
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(// add
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(
)"
		R"(// general-purpose kernel for addition of two tensors
)"
		R"(// pros: works for non-contiguous tensors, supports broadcast across dims 1, 2 and 3
)"
		R"(// cons: not very efficient
)"
		R"(kernel void kernel_add(
)"
		R"(        global char * src0,
)"
		R"(        ulong  offset0,
)"
		R"(        global char * src1,
)"
		R"(        ulong  offset1,
)"
		R"(        global char * dst,
)"
		R"(        ulong  offsetd,
)"
		R"(        int   ne00,
)"
		R"(        int   ne01,
)"
		R"(        int   ne02,
)"
		R"(        int   ne03,
)"
		R"(        ulong nb00,
)"
		R"(        ulong nb01,
)"
		R"(        ulong nb02,
)"
		R"(        ulong nb03,
)"
		R"(        int   ne10,
)"
		R"(        int   ne11,
)"
		R"(        int   ne12,
)"
		R"(        int   ne13,
)"
		R"(        ulong nb10,
)"
		R"(        ulong nb11,
)"
		R"(        ulong nb12,
)"
		R"(        ulong nb13,
)"
		R"(        int   ne0,
)"
		R"(        int   ne1,
)"
		R"(        int   ne2,
)"
		R"(        int   ne3,
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
		R"(    src1 = src1 + offset1;
)"
		R"(    dst = dst + offsetd;
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
		R"(    int i13 = i03 % ne13;
)"
		R"(    int i12 = i02 % ne12;
)"
		R"(    int i11 = i01 % ne11;
)"
		R"(
)"
		R"(    global char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
)"
		R"(    global char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
)"
		R"(    global char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;
)"
		R"(
)"
		R"(    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
)"
		R"(        const int i10 = i0 % ne10;
)"
		R"(        *((global float *)(dst_ptr + i0*nb0)) = *((global float *)(src0_ptr + i0*nb00)) + *((global float *)(src1_ptr + i10*nb10));
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(// assumption: src1 is a row
)"
		R"(// broadcast src1 into src0
)"
		R"(kernel void kernel_add_row(
)"
		R"(        global float4 * src0,
)"
		R"(        ulong  offset0,
)"
		R"(        global float4 * src1,
)"
		R"(        ulong  offset1,
)"
		R"(        global float4 * dst,
)"
		R"(        ulong  offsetd,
)"
		R"(        int ne
)"
		R"() {
)"
		R"(    src0 = (global float4*)((global char*)src0 + offset0);
)"
		R"(    src1 = (global float4*)((global char*)src1 + offset1);
)"
		R"(    dst = (global float4*)((global char*)dst + offsetd);
)"
		R"(
)"
		R"(    // This performs better than using %.
)"
		R"(    uint gid = get_global_id(0);
)"
		R"(    uint idx1 = gid - (gid/ne)*ne; // get_global_id(0) % ne
)"
		R"(    dst[gid] = src0[gid] + src1[idx1];
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_add_f16(
)"
		R"(        global char * src0,
)"
		R"(        ulong  offset0,
)"
		R"(        global char * src1,
)"
		R"(        ulong  offset1,
)"
		R"(        global char * dst,
)"
		R"(        ulong  offsetd,
)"
		R"(        int   ne00,
)"
		R"(        int   ne01,
)"
		R"(        int   ne02,
)"
		R"(        int   ne03,
)"
		R"(        ulong nb00,
)"
		R"(        ulong nb01,
)"
		R"(        ulong nb02,
)"
		R"(        ulong nb03,
)"
		R"(        int   ne10,
)"
		R"(        int   ne11,
)"
		R"(        int   ne12,
)"
		R"(        int   ne13,
)"
		R"(        ulong nb10,
)"
		R"(        ulong nb11,
)"
		R"(        ulong nb12,
)"
		R"(        ulong nb13,
)"
		R"(        int   ne0,
)"
		R"(        int   ne1,
)"
		R"(        int   ne2,
)"
		R"(        int   ne3,
)"
		R"(        ulong nb0,
)"
		R"(        ulong nb1,
)"
		R"(        ulong nb2,
)"
		R"(        ulong nb3,
)"
		R"(        int type_src0,
)"
		R"(        int type_src1
)"
		R"() {
)"
		R"(    src0 = src0 + offset0;
)"
		R"(    src1 = src1 + offset1;
)"
		R"(    dst = dst + offsetd;
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
		R"(    int i13 = i03 % ne13;
)"
		R"(    int i12 = i02 % ne12;
)"
		R"(    int i11 = i01 % ne11;
)"
		R"(
)"
		R"(    global char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
)"
		R"(    global char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
)"
		R"(    global char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;
)"
		R"(
)"
		R"(    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
)"
		R"(        const int i10 = i0 % ne10;
)"
		R"(
)"
		R"(        half v0, v1;
)"
		R"(        if (type_src0 == 1) {
)"
		R"(            v0 = convert_half(*((global float *)(src0_ptr + i0*nb00)));
)"
		R"(        } else {
)"
		R"(            v0 = *((global half *)(src0_ptr + i0*nb00));
)"
		R"(        }
)"
		R"(
)"
		R"(        if (type_src1 == 1) {
)"
		R"(            v1 = convert_half(*((global float *)(src1_ptr + i10*nb10)));
)"
		R"(        } else {
)"
		R"(            v1 = *((global half *)(src1_ptr + i10*nb10));
)"
		R"(        }
)"
		R"(
)"
		R"(        *((global half *)(dst_ptr + i0*nb0)) = v0 + v1;
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_add_row_f16(
)"
		R"(        global char * src0,
)"
		R"(        ulong  offset0,
)"
		R"(        global char * src1,
)"
		R"(        ulong  offset1,
)"
		R"(        global half4 * dst,
)"
		R"(        ulong  offsetd,
)"
		R"(        int ne,
)"
		R"(        int type_src0,
)"
		R"(        int type_src1
)"
		R"() {
)"
		R"(    dst = (global half4*)((global char*)dst + offsetd);
)"
		R"(
)"
		R"(    // This performs better than using %.
)"
		R"(    uint gid = get_global_id(0);
)"
		R"(    uint idx1 = gid - (gid/ne)*ne; // get_global_id(0) % ne
)"
		R"(
)"
		R"(    half4 v0, v1;
)"
		R"(    if (type_src0 == 1) {
)"
		R"(        global float4* src0_f32 = (global float4*)((global char*)src0 + offset0);
)"
		R"(        v0 = convert_half4(src0_f32[gid]);
)"
		R"(    } else {
)"
		R"(        global half4* src0_f16 = (global half4*)((global char*)src0 + offset0);
)"
		R"(        v0 = src0_f16[gid];
)"
		R"(    }
)"
		R"(
)"
		R"(    if (type_src1 == 1) {
)"
		R"(        global float4* src1_f32 = (global float4*)((global char*)src1 + offset1);
)"
		R"(        v1 = convert_half4(src1_f32[idx1]);
)"
		R"(    } else {
)"
		R"(        global half4* src1_f16 = (global half4*)((global char*)src1 + offset1);
)"
		R"(        v1 = src1_f16[idx1];
)"
		R"(    }
)"
		R"(
)"
		R"(    dst[gid] = v0 + v1;
)"
		R"(}
)"
