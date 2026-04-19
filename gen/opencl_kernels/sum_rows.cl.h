R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(#pragma OPENCL EXTENSION cl_khr_subgroups : enable
)"
R"(
)"
R"(// Most devices have max workgroup size of 1024, so this is enough for subgroup
)"
R"(// sizes of 16, 32, 64 and 128. Increase this value for smaller subgroups sizes
)"
R"(#define MAX_SUBGROUPS 64
)"
R"(kernel void kernel_sum_rows_f32(
)"
R"(    global char *  src0,
)"
R"(    ulong           offset0,
)"
R"(    global char *  dst,
)"
R"(    ulong           offsetd,
)"
R"(    int             ne00,
)"
R"(    int             ne01,
)"
R"(    int             ne02,
)"
R"(    int             ne03,
)"
R"(    ulong           nb01,
)"
R"(    ulong           nb02,
)"
R"(    ulong           nb03,
)"
R"(    ulong           nb1,
)"
R"(    ulong           nb2,
)"
R"(    ulong           nb3
)"
R"() {
)"
R"(    src0 = src0 + offset0;
)"
R"(    dst  = dst  + offsetd;
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
R"(    const int lid = get_local_id(0);
)"
R"(    const int lsize = get_local_size(0);
)"
R"(
)"
R"(    const uint sg_size = get_sub_group_size();
)"
R"(    const uint sg_id = get_sub_group_id();
)"
R"(    const uint sg_lid = get_sub_group_local_id();
)"
R"(
)"
R"(    __local float lmem[MAX_SUBGROUPS];
)"
R"(
)"
R"(    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
)"
R"(        return;
)"
R"(    }
)"
R"(
)"
R"(    if(sg_id == 0){
)"
R"(        lmem[sg_lid] = 0.0f;
)"
R"(    }
)"
R"(
)"
R"(    global float * src_row = (global float *) (src0 + i1*nb01 + i2*nb02 + i3*nb03);
)"
R"(    global float * dst_row = (global float *) (dst  + i1*nb1  + i2*nb2  + i3*nb3);
)"
R"(
)"
R"(    float sumf = 0.0f;
)"
R"(
)"
R"(    for (int i0 = lid; i0 < ne00; i0 += lsize) {
)"
R"(        sumf += src_row[i0];
)"
R"(    }
)"
R"(
)"
R"(    sumf = sub_group_reduce_add(sumf);
)"
R"(
)"
R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(
)"
R"(    if(sg_lid == 0){
)"
R"(        lmem[sg_id] = sumf;
)"
R"(    }
)"
R"(
)"
R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(
)"
R"(    sumf = lmem[sg_lid];
)"
R"(    sumf = sub_group_reduce_add(sumf);
)"
R"(
)"
R"(    if (lid == 0) {
)"
R"(        dst_row[0] = sumf;
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_sum_rows_f32_4(
)"
R"(    global char *  src0,
)"
R"(    ulong           offset0,
)"
R"(    global char *  dst,
)"
R"(    ulong           offsetd,
)"
R"(    int             ne00,
)"
R"(    int             ne01,
)"
R"(    int             ne02,
)"
R"(    int             ne03,
)"
R"(    ulong           nb01,
)"
R"(    ulong           nb02,
)"
R"(    ulong           nb03,
)"
R"(    ulong           nb1,
)"
R"(    ulong           nb2,
)"
R"(    ulong           nb3
)"
R"() {
)"
R"(    src0 = src0 + offset0;
)"
R"(    dst  = dst  + offsetd;
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
R"(    const int lid = get_local_id(0);
)"
R"(    const int lsize = get_local_size(0);
)"
R"(
)"
R"(    const uint sg_size = get_sub_group_size();
)"
R"(    const uint sg_id = get_sub_group_id();
)"
R"(    const uint sg_lid = get_sub_group_local_id();
)"
R"(
)"
R"(    __local float lmem[MAX_SUBGROUPS];
)"
R"(
)"
R"(    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
)"
R"(        return;
)"
R"(    }
)"
R"(
)"
R"(    if(sg_id == 0){
)"
R"(        lmem[sg_lid] = 0.0f;
)"
R"(    }
)"
R"(
)"
R"(    global float4 * src_row = (global float4 *) (src0 + i1*nb01 + i2*nb02 + i3*nb03);
)"
R"(    global float  * dst_row = (global float  *) (dst  + i1*nb1  + i2*nb2  + i3*nb3);
)"
R"(
)"
R"(    float4 sum_vec = (float4)0.0f;
)"
R"(
)"
R"(    for (int i0 = lid; i0 < ne00 / 4; i0 += lsize) {
)"
R"(        sum_vec += src_row[i0];
)"
R"(    }
)"
R"(
)"
R"(    float sumf = dot(sum_vec, (float4)(1.0f));
)"
R"(    sumf = sub_group_reduce_add(sumf);
)"
R"(
)"
R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(
)"
R"(    if(sg_lid == 0){
)"
R"(        lmem[sg_id] = sumf;
)"
R"(    }
)"
R"(
)"
R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(
)"
R"(    sumf = lmem[sg_lid];
)"
R"(    sumf = sub_group_reduce_add(sumf);
)"
R"(
)"
R"(    if (lid == 0) {
)"
R"(        dst_row[0] = sumf;
)"
R"(    }
)"
R"(}
)"
