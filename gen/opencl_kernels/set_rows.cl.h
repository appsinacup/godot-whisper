R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(
)"
R"(// v = { mp, L, d }
)"
R"(inline uint fastdiv(uint n, uint4 v) {
)"
R"(    uint msbs;
)"
R"(    msbs = mul_hi(n, v.s0);
)"
R"(    return (msbs + n) >> v.s1;
)"
R"(}
)"
R"(inline uint fastmod(uint n, uint4 v) {
)"
R"(    uint q = fastdiv(n, v);
)"
R"(    return n - q * v.s2;
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_set_rows_f32_i64(
)"
R"(        global char * src0,
)"
R"(        ulong         offset0,
)"
R"(        global char * src1,
)"
R"(        ulong         offset1,
)"
R"(        global char * dst,
)"
R"(        ulong         offsetd,
)"
R"(        int           ne01,
)"
R"(        ulong         nb01,
)"
R"(        ulong         nb02,
)"
R"(        ulong         nb03,
)"
R"(        uint4         ne11,
)"
R"(        uint4         ne12,
)"
R"(        ulong         nb10,
)"
R"(        ulong         nb11,
)"
R"(        ulong         nb12,
)"
R"(        int           nblk0,
)"
R"(        ulong         nb1,
)"
R"(        ulong         nb2,
)"
R"(        ulong         nb3
)"
R"() {
)"
R"(    src0 = src0 + offset0;
)"
R"(    src1 = src1 + offset1;
)"
R"(    dst  = dst  + offsetd;
)"
R"(
)"
R"(    int i03 = get_group_id(2);
)"
R"(    int i02 = get_group_id(1);
)"
R"(    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);
)"
R"(
)"
R"(    if (i01 >= ne01) {
)"
R"(        return;
)"
R"(    }
)"
R"(
)"
R"(    //int i12 = i03%ne12;
)"
R"(    //int i11 = i02%ne11;
)"
R"(    int i12 = fastmod(i03, ne12);
)"
R"(    int i11 = fastmod(i02, ne11);
)"
R"(
)"
R"(    int i10 = i01;
)"
R"(    long i1 = ((global long *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];
)"
R"(
)"
R"(    global float * dst_row = (global float *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
)"
R"(    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);
)"
R"(
)"
R"(    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
)"
R"(        dst_row[ind] = (float)src_row[ind];
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_set_rows_f16_i64(
)"
R"(        global char * src0,
)"
R"(        ulong         offset0,
)"
R"(        global char * src1,
)"
R"(        ulong         offset1,
)"
R"(        global char * dst,
)"
R"(        ulong         offsetd,
)"
R"(        int           ne01,
)"
R"(        ulong         nb01,
)"
R"(        ulong         nb02,
)"
R"(        ulong         nb03,
)"
R"(        uint4         ne11,
)"
R"(        uint4         ne12,
)"
R"(        ulong         nb10,
)"
R"(        ulong         nb11,
)"
R"(        ulong         nb12,
)"
R"(        int           nblk0,
)"
R"(        ulong         nb1,
)"
R"(        ulong         nb2,
)"
R"(        ulong         nb3
)"
R"() {
)"
R"(    src0 = src0 + offset0;
)"
R"(    src1 = src1 + offset1;
)"
R"(    dst  = dst  + offsetd;
)"
R"(
)"
R"(    int i03 = get_group_id(2);
)"
R"(    int i02 = get_group_id(1);
)"
R"(    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);
)"
R"(
)"
R"(    if (i01 >= ne01) {
)"
R"(        return;
)"
R"(    }
)"
R"(
)"
R"(    //int i12 = i03%ne12;
)"
R"(    //int i11 = i02%ne11;
)"
R"(    int i12 = fastmod(i03, ne12);
)"
R"(    int i11 = fastmod(i02, ne11);
)"
R"(
)"
R"(    int i10 = i01;
)"
R"(    long i1 = ((global long *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];
)"
R"(
)"
R"(    global half  * dst_row = (global half  *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
)"
R"(    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);
)"
R"(
)"
R"(    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
)"
R"(        dst_row[ind] = src_row[ind];
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_set_rows_f32_i32(
)"
R"(        global char * src0,
)"
R"(        ulong         offset0,
)"
R"(        global char * src1,
)"
R"(        ulong         offset1,
)"
R"(        global char * dst,
)"
R"(        ulong         offsetd,
)"
R"(        int           ne01,
)"
R"(        ulong         nb01,
)"
R"(        ulong         nb02,
)"
R"(        ulong         nb03,
)"
R"(        uint4         ne11,
)"
R"(        uint4         ne12,
)"
R"(        ulong         nb10,
)"
R"(        ulong         nb11,
)"
R"(        ulong         nb12,
)"
R"(        int           nblk0,
)"
R"(        ulong         nb1,
)"
R"(        ulong         nb2,
)"
R"(        ulong         nb3
)"
R"() {
)"
R"(    src0 = src0 + offset0;
)"
R"(    src1 = src1 + offset1;
)"
R"(    dst  = dst  + offsetd;
)"
R"(
)"
R"(    int i03 = get_group_id(2);
)"
R"(    int i02 = get_group_id(1);
)"
R"(    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);
)"
R"(
)"
R"(    if (i01 >= ne01) {
)"
R"(        return;
)"
R"(    }
)"
R"(
)"
R"(    //int i12 = i03%ne12;
)"
R"(    //int i11 = i02%ne11;
)"
R"(    int i12 = fastmod(i03, ne12);
)"
R"(    int i11 = fastmod(i02, ne11);
)"
R"(
)"
R"(    int i10 = i01;
)"
R"(    int i1  = ((global int *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];
)"
R"(
)"
R"(    global float * dst_row = (global float *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
)"
R"(    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);
)"
R"(
)"
R"(    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
)"
R"(        dst_row[ind] = (float)src_row[ind];
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_set_rows_f16_i32(
)"
R"(        global char * src0,
)"
R"(        ulong         offset0,
)"
R"(        global char * src1,
)"
R"(        ulong         offset1,
)"
R"(        global char * dst,
)"
R"(        ulong         offsetd,
)"
R"(        int           ne01,
)"
R"(        ulong         nb01,
)"
R"(        ulong         nb02,
)"
R"(        ulong         nb03,
)"
R"(        uint4         ne11,
)"
R"(        uint4         ne12,
)"
R"(        ulong         nb10,
)"
R"(        ulong         nb11,
)"
R"(        ulong         nb12,
)"
R"(        int           nblk0,
)"
R"(        ulong         nb1,
)"
R"(        ulong         nb2,
)"
R"(        ulong         nb3
)"
R"() {
)"
R"(    src0 = src0 + offset0;
)"
R"(    src1 = src1 + offset1;
)"
R"(    dst  = dst  + offsetd;
)"
R"(
)"
R"(    int i03 = get_group_id(2);
)"
R"(    int i02 = get_group_id(1);
)"
R"(    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);
)"
R"(
)"
R"(    if (i01 >= ne01) {
)"
R"(        return;
)"
R"(    }
)"
R"(
)"
R"(    //int i12 = i03%ne12;
)"
R"(    //int i11 = i02%ne11;
)"
R"(    int i12 = fastmod(i03, ne12);
)"
R"(    int i11 = fastmod(i02, ne11);
)"
R"(
)"
R"(    int i10 = i01;
)"
R"(    int i1  = ((global int *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];
)"
R"(
)"
R"(    global half  * dst_row = (global half  *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
)"
R"(    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);
)"
R"(
)"
R"(    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
)"
R"(        dst_row[ind] = src_row[ind];
)"
R"(    }
)"
R"(}
)"
