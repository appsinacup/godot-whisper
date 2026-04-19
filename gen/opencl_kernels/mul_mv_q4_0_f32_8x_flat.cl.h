R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(
)"
R"(#ifdef cl_intel_subgroups
)"
R"(#pragma OPENCL EXTENSION cl_intel_subgroups : enable
)"
R"(#else
)"
R"(#pragma OPENCL EXTENSION cl_khr_subgroups : enable
)"
R"(#endif
)"
R"(
)"
R"(#ifdef cl_intel_required_subgroup_size
)"
R"(#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
)"
R"(#define INTEL_GPU 1
)"
R"(#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
)"
R"(#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
)"
R"(#elif defined(cl_qcom_reqd_sub_group_size)
)"
R"(#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
)"
R"(#define ADRENO_GPU 1
)"
R"(#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
)"
R"(#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
)"
R"(#endif
)"
R"(
)"
R"(#define QK4_0                   32
)"
R"(#define QR4_0                   2
)"
R"(#define QK4_1                   32
)"
R"(#define QR4_1                   2
)"
R"(#define QK5_0                   32
)"
R"(#define QR5_0                   2
)"
R"(#define QK5_1                   32
)"
R"(#define QR5_1                   2
)"
R"(#define QK8_0                   32
)"
R"(#define QR8_0                   1
)"
R"(#define QK_K                    256
)"
R"(#define K_QUANTS_PER_ITERATION  2
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
R"(// This function requires the original shuffled weights.
)"
R"(// As a reminder, the original weights are shuffled so that (q[0], q[16]) are
)"
R"(// packed together in a byte, so are (q[1], q[17]) and so on.
)"
R"(inline float block_q_4_0_dot_y_flat(
)"
R"(        global uchar * x,
)"
R"(        global half  * dh,
)"
R"(        float sumy,
)"
R"(        float16 yl,
)"
R"(        int il
)"
R"() {
)"
R"(    float           d   = *dh;
)"
R"(    global ushort * qs  = ((global ushort *)x + il/2);
)"
R"(    float           acc = 0.f;
)"
R"(
)"
R"(    acc += yl.s0 * (qs[0] & 0x000F);
)"
R"(    acc += yl.s1 * (qs[0] & 0x0F00);
)"
R"(    acc += yl.s8 * (qs[0] & 0x00F0);
)"
R"(    acc += yl.s9 * (qs[0] & 0xF000);
)"
R"(
)"
R"(    acc += yl.s2 * (qs[1] & 0x000F);
)"
R"(    acc += yl.s3 * (qs[1] & 0x0F00);
)"
R"(    acc += yl.sa * (qs[1] & 0x00F0);
)"
R"(    acc += yl.sb * (qs[1] & 0xF000);
)"
R"(
)"
R"(    acc += yl.s4 * (qs[2] & 0x000F);
)"
R"(    acc += yl.s5 * (qs[2] & 0x0F00);
)"
R"(    acc += yl.sc * (qs[2] & 0x00F0);
)"
R"(    acc += yl.sd * (qs[2] & 0xF000);
)"
R"(
)"
R"(    acc += yl.s6 * (qs[3] & 0x000F);
)"
R"(    acc += yl.s7 * (qs[3] & 0x0F00);
)"
R"(    acc += yl.se * (qs[3] & 0x00F0);
)"
R"(    acc += yl.sf * (qs[3] & 0xF000);
)"
R"(
)"
R"(    return d * (sumy * -8.f + acc);
)"
R"(}
)"
R"(
)"
R"(//
)"
R"(// This variant outputs 8 values.
)"
R"(//
)"
R"(#undef N_DST
)"
R"(#undef N_SIMDGROUP
)"
R"(#undef N_SIMDWIDTH
)"
R"(
)"
R"(#ifdef INTEL_GPU
)"
R"(#define N_DST 8 // each SIMD group works on 8 rows
)"
R"(#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
)"
R"(#define N_SIMDWIDTH 16 // assuming SIMD group size is 32
)"
R"(#elif defined (ADRENO_GPU)
)"
R"(#define N_DST 8
)"
R"(#define N_SIMDGROUP 1
)"
R"(#define N_SIMDWIDTH 64
)"
R"(#endif
)"
R"(
)"
R"(inline void mul_vec_q_n_f32_8x_flat(
)"
R"(        global uchar * src0_q,
)"
R"(        global half  * src0_d,
)"
R"(        global float * src1,
)"
R"(        global float * dst,
)"
R"(        int ne00,
)"
R"(        int ne01,
)"
R"(        int ne02,
)"
R"(        int ne10,
)"
R"(        int ne12,
)"
R"(        int ne0,
)"
R"(        int ne1,
)"
R"(        int r2,
)"
R"(        int r3
)"
R"() {
)"
R"(    const ulong nb = ne00/QK4_0;
)"
R"(
)"
R"(    int r0 = get_group_id(0);
)"
R"(    int r1 = get_group_id(1);
)"
R"(    int im = get_group_id(2);
)"
R"(
)"
R"(    // (r0 * N_SIMDGROUP + get_sub_group_id()) is the linear global id of
)"
R"(    // a SIMD group in the grid. Each SIMD group produces N_DST values in the
)"
R"(    // result, hence uses nb blocks, i.e., the offset becomes first_row*nb.
)"
R"(    // Currently with llama2 7B, im is always 0.
)"
R"(    // TODO: how to handle im/gqa*(nb*ne0)?
)"
R"(    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;
)"
R"(
)"
R"(    int i12 = im%ne12;
)"
R"(    int i13 = im/ne12;
)"
R"(
)"
R"(    // The number of scales is the same as the number of blocks.
)"
R"(    ulong offset0_d = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
)"
R"(    // Each block contains QK4_0/2 uchars, hence offset for qs is as follows.
)"
R"(    ulong offset0_q = (first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02)) * QK4_0/2;
)"
R"(
)"
R"(    global uchar * x = (global uchar *) src0_q + offset0_q;
)"
R"(    global half  * d = (global half  *) src0_d + offset0_d;
)"
R"(    global float * y = (global float *) src1   + r1*ne10 + im*ne00*ne1;
)"
R"(
)"
R"(    float16 yl;
)"
R"(    float8 sumf = 0.f;
)"
R"(
)"
R"(    int ix = get_sub_group_local_id()/2;
)"
R"(    int il = 8*(get_sub_group_local_id()%2);
)"
R"(
)"
R"(    global float * yb = y + ix*QK4_0 + il;
)"
R"(
)"
R"(    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/2) {
)"
R"(        float sumy = 0.f;
)"
R"(
)"
R"(        sumy += yb[0];
)"
R"(        sumy += yb[1];
)"
R"(        sumy += yb[2];
)"
R"(        sumy += yb[3];
)"
R"(        sumy += yb[4];
)"
R"(        sumy += yb[5];
)"
R"(        sumy += yb[6];
)"
R"(        sumy += yb[7];
)"
R"(
)"
R"(        sumy += yb[16];
)"
R"(        sumy += yb[17];
)"
R"(        sumy += yb[18];
)"
R"(        sumy += yb[19];
)"
R"(        sumy += yb[20];
)"
R"(        sumy += yb[21];
)"
R"(        sumy += yb[22];
)"
R"(        sumy += yb[23];
)"
R"(
)"
R"(        yl.s0 = yb[0];
)"
R"(        yl.s1 = yb[1]/256.f;
)"
R"(
)"
R"(        yl.s2 = yb[2];
)"
R"(        yl.s3 = yb[3]/256.f;
)"
R"(
)"
R"(        yl.s4 = yb[4];
)"
R"(        yl.s5 = yb[5]/256.f;
)"
R"(
)"
R"(        yl.s6 = yb[6];
)"
R"(        yl.s7 = yb[7]/256.f;
)"
R"(
)"
R"(        yl.s8 = yb[16]/16.f;
)"
R"(        yl.s9 = yb[17]/4096.f;
)"
R"(
)"
R"(        yl.sa = yb[18]/16.f;
)"
R"(        yl.sb = yb[19]/4096.f;
)"
R"(
)"
R"(        yl.sc = yb[20]/16.f;
)"
R"(        yl.sd = yb[21]/4096.f;
)"
R"(
)"
R"(        yl.se = yb[22]/16.f;
)"
R"(        yl.sf = yb[23]/4096.f;
)"
R"(
)"
R"(        sumf.s0 += block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 0*nb*QK4_0/2, d + ib + 0*nb, sumy, yl, il);
)"
R"(        sumf.s1 += block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 1*nb*QK4_0/2, d + ib + 1*nb, sumy, yl, il);
)"
R"(        sumf.s2 += block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 2*nb*QK4_0/2, d + ib + 2*nb, sumy, yl, il);
)"
R"(        sumf.s3 += block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 3*nb*QK4_0/2, d + ib + 3*nb, sumy, yl, il);
)"
R"(
)"
R"(        sumf.s4 += block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 4*nb*QK4_0/2, d + ib + 4*nb, sumy, yl, il);
)"
R"(        sumf.s5 += block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 5*nb*QK4_0/2, d + ib + 5*nb, sumy, yl, il);
)"
R"(        sumf.s6 += block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 6*nb*QK4_0/2, d + ib + 6*nb, sumy, yl, il);
)"
R"(        sumf.s7 += block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 7*nb*QK4_0/2, d + ib + 7*nb, sumy, yl, il);
)"
R"(
)"
R"(        yb += QK4_0 * (N_SIMDWIDTH/2);
)"
R"(    }
)"
R"(
)"
R"(    float8 tot = (float8)(
)"
R"(        sub_group_reduce_add(sumf.s0), sub_group_reduce_add(sumf.s1),
)"
R"(        sub_group_reduce_add(sumf.s2), sub_group_reduce_add(sumf.s3),
)"
R"(        sub_group_reduce_add(sumf.s4), sub_group_reduce_add(sumf.s5),
)"
R"(        sub_group_reduce_add(sumf.s6), sub_group_reduce_add(sumf.s7)
)"
R"(    );
)"
R"(
)"
R"(    if (get_sub_group_local_id() == 0) {
)"
R"(        if (first_row + 0 < ne01) {
)"
R"(            dst[r1*ne0 + im*ne0*ne1 + first_row + 0] = tot.s0;
)"
R"(        }
)"
R"(        if (first_row + 1 < ne01) {
)"
R"(            dst[r1*ne0 + im*ne0*ne1 + first_row + 1] = tot.s1;
)"
R"(        }
)"
R"(        if (first_row + 2 < ne01) {
)"
R"(            dst[r1*ne0 + im*ne0*ne1 + first_row + 2] = tot.s2;
)"
R"(        }
)"
R"(        if (first_row + 3 < ne01) {
)"
R"(            dst[r1*ne0 + im*ne0*ne1 + first_row + 3] = tot.s3;
)"
R"(        }
)"
R"(
)"
R"(        if (first_row + 4 < ne01) {
)"
R"(            dst[r1*ne0 + im*ne0*ne1 + first_row + 4] = tot.s4;
)"
R"(        }
)"
R"(        if (first_row + 5 < ne01) {
)"
R"(            dst[r1*ne0 + im*ne0*ne1 + first_row + 5] = tot.s5;
)"
R"(        }
)"
R"(        if (first_row + 6 < ne01) {
)"
R"(            dst[r1*ne0 + im*ne0*ne1 + first_row + 6] = tot.s6;
)"
R"(        }
)"
R"(        if (first_row + 7 < ne01) {
)"
R"(            dst[r1*ne0 + im*ne0*ne1 + first_row + 7] = tot.s7;
)"
R"(        }
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(#ifdef INTEL_GPU
)"
R"(REQD_SUBGROUP_SIZE_16
)"
R"(#elif defined (ADRENO_GPU)
)"
R"(REQD_SUBGROUP_SIZE_64
)"
R"(#endif
)"
R"(kernel void kernel_mul_mat_q4_0_f32_8x_flat(
)"
R"(        global uchar * src0_q,
)"
R"(        global half  * src0_d,
)"
R"(        global float * src1,
)"
R"(        ulong offset1,
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
R"(        int ne10,
)"
R"(        int ne12,
)"
R"(        int ne0,
)"
R"(        int ne1,
)"
R"(        int r2,
)"
R"(        int r3
)"
R"() {
)"
R"(    src1 = (global float*)((global char*)src1 + offset1);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    mul_vec_q_n_f32_8x_flat(src0_q, src0_d, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3);
)"
R"(}
)"
