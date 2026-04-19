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
R"(//------------------------------------------------------------------------------
)"
R"(// mul_vec_q_n_f32
)"
R"(//------------------------------------------------------------------------------
)"
R"(// function for calculate inner product between half a q4_0 block and 16 floats (yl), sumy is SUM(yl[i])
)"
R"(// il indicates where the q4 quants begin (0 or QK4_0/4)
)"
R"(// we assume that the yl's have been multiplied with the appropriate scale factor
)"
R"(// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
)"
R"(inline float block_q_4_0_dot_y(
)"
R"(        global struct block_q4_0 * qb_curr,
)"
R"(        float sumy,
)"
R"(        private float * yl,
)"
R"(        int il
)"
R"() {
)"
R"(    float d = qb_curr->d;
)"
R"(    float2 acc = 0.f;
)"
R"(    global ushort * qs = ((global ushort *)qb_curr + 1 + il/2);
)"
R"(    for (int i = 0; i < 8; i+=2) {
)"
R"(        acc.s0 += yl[i + 0] * (qs[i / 2] & 0x000F)
)"
R"(                + yl[i + 1] * (qs[i / 2] & 0x0F00);
)"
R"(        acc.s1 += yl[i + 8] * (qs[i / 2] & 0x00F0)
)"
R"(                + yl[i + 9] * (qs[i / 2] & 0xF000);
)"
R"(    }
)"
R"(    return d * (sumy * -8.f + acc.s0 + acc.s1);
)"
R"(}
)"
R"(
)"
R"(#ifdef INTEL_GPU
)"
R"(#define N_DST 4 // each SIMD group works on 4 rows
)"
R"(#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
)"
R"(#define N_SIMDWIDTH 16 // assuming SIMD group size is 16
)"
R"(#elif defined (ADRENO_GPU)
)"
R"(#define N_DST 4
)"
R"(#define N_SIMDGROUP 1
)"
R"(#define N_SIMDWIDTH 64
)"
R"(#endif
)"
R"(
)"
R"(inline void mul_vec_q_n_f32(
)"
R"(        global void * src0,
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
R"(
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
R"(    // (r0 * N_SIMDGROUP + get_sub_group_id()) is essenatially the linear global
)"
R"(    // id of a SIMD group in the grid.
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
R"(    ulong offset0 = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
)"
R"(
)"
R"(    global struct block_q4_0 * x = (global struct block_q4_0 *) src0 + offset0;
)"
R"(    global float             * y = (global float             *) src1 + r1*ne10 + im*ne00*ne1;
)"
R"(
)"
R"(    float yl[16];       // src1 vector cache
)"
R"(    float sumf[N_DST]={0.f};
)"
R"(
)"
R"(    int ix = get_sub_group_local_id()/2;
)"
R"(    int il = 8*(get_sub_group_local_id()%2);
)"
R"(
)"
R"(    global float * yb = y + ix * QK4_0 + il;
)"
R"(
)"
R"(    // each thread in a SIMD group deals with half a block.
)"
R"(    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/2) {
)"
R"(        float sumy = 0;
)"
R"(        for (int i = 0; i < 8; i += 2) {
)"
R"(            sumy += yb[i] + yb[i+1];
)"
R"(            yl[i+0] = yb[i+ 0];
)"
R"(            yl[i+1] = yb[i+ 1]/256.f;
)"
R"(            sumy += yb[i+16] + yb[i+17];
)"
R"(            yl[i+8] = yb[i+16]/16.f;
)"
R"(            yl[i+9] = yb[i+17]/4096.f;
)"
R"(        }
)"
R"(
)"
R"(        for (int row = 0; row < N_DST; row++) {
)"
R"(            sumf[row] += block_q_4_0_dot_y(x+ib+row*nb, sumy, yl, il);
)"
R"(        }
)"
R"(
)"
R"(        // One thread in a SIMD group (i.e., subgroup) handles a half block,
)"
R"(        // hence then entire SIMD group handles SIMDWIDTH/2 blocks.
)"
R"(        // y points to the activation matrix (of type float). Therefore for
)"
R"(        // one thread, the # of blocks y should advance is SIMDWIDTH/2 (because
)"
R"(        // SIMDWIDTH/2 blocks are processed by a SIMD group) - in terms of
)"
R"(        // floats, it is QK4_0 * (SIMDWIDTH/2), where QK4_0 is the block size.
)"
R"(        yb += QK4_0 * (N_SIMDWIDTH/2);
)"
R"(    }
)"
R"(
)"
R"(    // The above does not work for Adreno - it produces incorrect results for
)"
R"(    // row = 1, 2, 3 and only row = 0 gives the correct result.
)"
R"(    // If N_DST is changed, the below array must be initialized accordingly.
)"
R"(    // This also seems to perform better on Intel.
)"
R"(    float tot[N_DST] = {
)"
R"(        sub_group_reduce_add(sumf[0]), sub_group_reduce_add(sumf[1]),
)"
R"(        sub_group_reduce_add(sumf[2]), sub_group_reduce_add(sumf[3])};
)"
R"(    for (int row = 0; row < N_DST; ++row) {
)"
R"(        if (get_sub_group_local_id() == 0 && first_row + row < ne01) {
)"
R"(            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = tot[row];
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
R"(kernel void kernel_mul_mat_q4_0_f32(
)"
R"(        global void * src0,
)"
R"(        ulong offset0,
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
R"(    src0 = (global void*)((global char*)src0 + offset0);
)"
R"(    src1 = (global float*)((global char*)src1 + offset1);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    mul_vec_q_n_f32(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3);
)"
R"(}
)"
