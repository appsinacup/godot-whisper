R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(
)"
R"(typedef char int8_t;
)"
R"(typedef uchar uint8_t;
)"
R"(typedef short int16_t;
)"
R"(typedef ushort uint16_t;
)"
R"(typedef int int32_t;
)"
R"(typedef uint uint32_t;
)"
R"(
)"
R"(#define QK4_0                   32
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// block_q4_0
)"
R"(//------------------------------------------------------------------------------
)"
R"(struct block_q4_0
)"
R"({
)"
R"(    half d;
)"
R"(    uint8_t qs[QK4_0 / 2];
)"
R"(};
)"
R"(
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// dequantize_q4_0_f32, dequantize_q4_0_f16
)"
R"(//------------------------------------------------------------------------------
)"
R"(void dequantize_q4_0_f32(global struct block_q4_0 * xb, short il, float16 * reg) {
)"
R"(    global ushort * qs = ((global ushort *)xb + 1);
)"
R"(    float d1 = il ? (xb->d / 16.h) : xb->d;
)"
R"(    float d2 = d1 / 256.f;
)"
R"(    float md = -8.h * xb->d;
)"
R"(    ushort mask0 = il ? 0x00F0 : 0x000F;
)"
R"(    ushort mask1 = mask0 << 8;
)"
R"(
)"
R"(    reg->s0 = d1 * (qs[0] & mask0) + md;
)"
R"(    reg->s1 = d2 * (qs[0] & mask1) + md;
)"
R"(
)"
R"(    reg->s2 = d1 * (qs[1] & mask0) + md;
)"
R"(    reg->s3 = d2 * (qs[1] & mask1) + md;
)"
R"(
)"
R"(    reg->s4 = d1 * (qs[2] & mask0) + md;
)"
R"(    reg->s5 = d2 * (qs[2] & mask1) + md;
)"
R"(
)"
R"(    reg->s6 = d1 * (qs[3] & mask0) + md;
)"
R"(    reg->s7 = d2 * (qs[3] & mask1) + md;
)"
R"(
)"
R"(    reg->s8 = d1 * (qs[4] & mask0) + md;
)"
R"(    reg->s9 = d2 * (qs[4] & mask1) + md;
)"
R"(
)"
R"(    reg->sa = d1 * (qs[5] & mask0) + md;
)"
R"(    reg->sb = d2 * (qs[5] & mask1) + md;
)"
R"(
)"
R"(    reg->sc = d1 * (qs[6] & mask0) + md;
)"
R"(    reg->sd = d2 * (qs[6] & mask1) + md;
)"
R"(
)"
R"(    reg->se = d1 * (qs[7] & mask0) + md;
)"
R"(    reg->sf = d2 * (qs[7] & mask1) + md;
)"
R"(}
)"
R"(
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// get_rows
)"
R"(//------------------------------------------------------------------------------
)"
R"(kernel void kernel_get_rows_f32(
)"
R"(        global void * src0,
)"
R"(        ulong offset0,
)"
R"(        global int * src1,
)"
R"(        ulong offset1,
)"
R"(        global float * dst,
)"
R"(        ulong offsetd,
)"
R"(        int ne00,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        int ne10,
)"
R"(        ulong nb10,
)"
R"(        ulong nb11,
)"
R"(        ulong nb12,
)"
R"(        ulong nb1,
)"
R"(        ulong nb2,
)"
R"(        ulong nb3
)"
R"() {
)"
R"(    src0 = (global void*)((global char*)src0 + offset0);
)"
R"(    src1 = (global int*)((global char*)src1 + offset1);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    int i10 = get_group_id(0);
)"
R"(    int i11 = get_group_id(1);
)"
R"(    int i12 = get_group_id(2);
)"
R"(
)"
R"(    int r = ((global int *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];
)"
R"(
)"
R"(    int i02 = i11;
)"
R"(    int i03 = i12;
)"
R"(
)"
R"(    for (int ind = get_local_id(0); ind < ne00; ind += get_local_size(0)) {
)"
R"(        if (ind >= ne00) {
)"
R"(            return;
)"
R"(        }
)"
R"(        ((global float *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1))[ind] =
)"
R"(            ((global float *) ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03))[ind];
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_get_rows_f16(
)"
R"(        global void * src0,
)"
R"(        ulong offset0,
)"
R"(        global int * src1,
)"
R"(        ulong offset1,
)"
R"(        global float * dst,
)"
R"(        ulong offsetd,
)"
R"(        int ne00,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        int ne10,
)"
R"(        ulong nb10,
)"
R"(        ulong nb11,
)"
R"(        ulong nb12,
)"
R"(        ulong nb1,
)"
R"(        ulong nb2,
)"
R"(        ulong nb3
)"
R"() {
)"
R"(    src0 = (global void*)((global char*)src0 + offset0);
)"
R"(    src1 = (global int*)((global char*)src1 + offset1);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    int i10 = get_group_id(0);
)"
R"(    int i11 = get_group_id(1);
)"
R"(    int i12 = get_group_id(2);
)"
R"(
)"
R"(    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];
)"
R"(
)"
R"(    int i02 = i11;
)"
R"(    int i03 = i12;
)"
R"(
)"
R"(    for (int ind = get_local_id(0); ind < ne00; ind += get_local_size(0)) {
)"
R"(        if (ind >= ne00) {
)"
R"(            return;
)"
R"(        }
)"
R"(        ((global float *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1))[ind] =
)"
R"(            ((global half *) ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03))[ind];
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_get_rows_q4_0(
)"
R"(        global void * src0,
)"
R"(        ulong offset0,
)"
R"(        global int * src1,
)"
R"(        ulong offset1,
)"
R"(        global float * dst,
)"
R"(        ulong offsetd,
)"
R"(        int ne00,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        int ne10,
)"
R"(        ulong nb10,
)"
R"(        ulong nb11,
)"
R"(        ulong nb12,
)"
R"(        ulong nb1,
)"
R"(        ulong nb2,
)"
R"(        ulong nb3
)"
R"() {
)"
R"(    src0 = (global void*)((global char*)src0 + offset0);
)"
R"(    src1 = (global int*)((global char*)src1 + offset1);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    const int NL = 2;
)"
R"(
)"
R"(    int i10 = get_group_id(0);
)"
R"(    int i11 = get_group_id(1);
)"
R"(    int i12 = get_group_id(2);
)"
R"(
)"
R"(    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];
)"
R"(
)"
R"(    int i02 = i11;
)"
R"(    int i03 = i12;
)"
R"(
)"
R"(    for (int ind = get_local_id(0); ind < ne00/16; ind += get_local_size(0)) {
)"
R"(        float16 temp;
)"
R"(        if (ind >= ne00) {
)"
R"(            return;
)"
R"(        }
)"
R"(        dequantize_q4_0_f32(
)"
R"(            ((global struct block_q4_0 *) ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03)) + ind/NL, ind%NL, &temp);
)"
R"(        *(((global float16 *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1)) + ind) = temp;
)"
R"(    }
)"
R"(}
)"
