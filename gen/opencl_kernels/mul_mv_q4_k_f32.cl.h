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
		R"(//------------------------------------------------------------------------------
)"
		R"(// block_q4_K
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(#define QK_K            256
)"
		R"(#define K_SCALE_SIZE    12
)"
		R"(
)"
		R"(// 8 blocks of 32 elements each
)"
		R"(// weight is represented as x = a * q + b
)"
		R"(typedef struct {
)"
		R"(    half d;    // super-block scale for quantized scales
)"
		R"(    half dmin; // super-block scale for quantized mins
)"
		R"(
)"
		R"(    uchar scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
)"
		R"(    uchar qs[QK_K/2];           // 4-bit quants
)"
		R"(} block_q4_K;
)"
		R"(
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
		R"(#define N_DST 4 // number of rows each SIMD group works on
)"
		R"(#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
)"
		R"(#define N_SIMDWIDTH 16 // SIMD group size
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
		R"(#undef  BLOCK_STRIDE
)"
		R"(// number of (super) blocks each subgroup processes
)"
		R"(// each thread in a subgroup processes a block (32 weights)
)"
		R"(#define BLOCK_STRIDE (N_SIMDWIDTH/8)
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
		R"(kernel void kernel_mul_mv_q4_K_f32(
)"
		R"(        global char * src0,
)"
		R"(        int offset0,
)"
		R"(        global char * src1,
)"
		R"(        int offset1,
)"
		R"(        global char * dst,
)"
		R"(        int offsetd,
)"
		R"(        int ne00,
)"
		R"(        int ne01,
)"
		R"(        ulong nb01,
)"
		R"(        ulong nb02,
)"
		R"(        ulong nb03,
)"
		R"(        int ne12,
)"
		R"(        ulong nb11,
)"
		R"(        ulong nb12,
)"
		R"(        ulong nb13,
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
		R"(    src0 = src0 + offset0;
)"
		R"(    src1 = src1 + offset1;
)"
		R"(    dst  = dst  + offsetd;
)"
		R"(
)"
		R"(    ushort kmask1 = 0x3f3f;
)"
		R"(    ushort kmask2 = 0x0f0f;
)"
		R"(    ushort kmask3 = 0xc0c0;
)"
		R"(
)"
		R"(    int ix = get_sub_group_local_id()/8;  // super block index
)"
		R"(    int it = get_sub_group_local_id()%8;  // block index (inside super block)
)"
		R"(    int iq = it/4;     // 0 or 1 - first or second half of the super block
)"
		R"(    int ir = it%4;     // 0...3 - block index in the half super block
)"
		R"(
)"
		R"(    int nb = ne00/QK_K;
)"
		R"(
)"
		R"(    int r0 = get_group_id(0);
)"
		R"(    int r1 = get_group_id(1);
)"
		R"(    int im = get_group_id(2);
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
		R"(    int offset_src0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
)"
		R"(    int offset_src1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;
)"
		R"(
)"
		R"(    global block_q4_K * x = (global block_q4_K *) (src0 + offset_src0);
)"
		R"(    global float      * y = (global float      *) (src1 + offset_src1);
)"
		R"(
)"
		R"(    float yl[16];
)"
		R"(    float yh[16];
)"
		R"(    float sumf[N_DST] = {0.f};
)"
		R"(    float all_sum;
)"
		R"(
)"
		R"(    global float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;
)"
		R"(
)"
		R"(    ushort  sc16[4];
)"
		R"(    uchar * sc8 = (uchar *)sc16;
)"
		R"(
)"
		R"(    for (int ib = ix; ib < nb; ib += BLOCK_STRIDE) {
)"
		R"(        float4 sumy = {0.f, 0.f, 0.f, 0.f};
)"
		R"(        for (int i = 0; i < 8; ++i) {
)"
		R"(            yl[i+0] = y4[i+0];
)"
		R"(            sumy.s0 += yl[i+0];
)"
		R"(
)"
		R"(            yl[i+8] = y4[i+32];
)"
		R"(            sumy.s1 += yl[i+8];
)"
		R"(
)"
		R"(            yh[i+0] = y4[i+128];
)"
		R"(            sumy.s2 += yh[i+0];
)"
		R"(
)"
		R"(            yh[i+8] = y4[i+160];
)"
		R"(            sumy.s3 += yh[i+8];
)"
		R"(        }
)"
		R"(
)"
		R"(        global ushort * sc = (global ushort *)x[ib].scales + iq;
)"
		R"(        global ushort * q1 = (global ushort *)x[ib].qs + 16 * iq + 4 * ir;
)"
		R"(        global half     * dh = &x[ib].d;
)"
		R"(
)"
		R"(        for (int row = 0; row < N_DST; row++) {
)"
		R"(            sc16[0] = sc[0] & kmask1;
)"
		R"(            sc16[1] = sc[2] & kmask1;
)"
		R"(            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
)"
		R"(            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
)"
		R"(
)"
		R"(            global ushort * q2 = q1 + 32;
)"
		R"(
)"
		R"(            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
)"
		R"(            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
)"
		R"(            for (int i = 0; i < 8; i += 2) {
)"
		R"(                acc1.s0 += yl[i+0] * (q1[i/2] & 0x000F);
)"
		R"(                acc1.s1 += yl[i+1] * (q1[i/2] & 0x0F00);
)"
		R"(                acc1.s2 += yl[i+8] * (q1[i/2] & 0x00F0);
)"
		R"(                acc1.s3 += yl[i+9] * (q1[i/2] & 0xF000);
)"
		R"(                acc2.s0 += yh[i+0] * (q2[i/2] & 0x000F);
)"
		R"(                acc2.s1 += yh[i+1] * (q2[i/2] & 0x0F00);
)"
		R"(                acc2.s2 += yh[i+8] * (q2[i/2] & 0x00F0);
)"
		R"(                acc2.s3 += yh[i+9] * (q2[i/2] & 0xF000);
)"
		R"(            }
)"
		R"(
)"
		R"(            float dall = dh[0];
)"
		R"(            float dmin = dh[1];
)"
		R"(            sumf[row] += dall * ((acc1.s0 + 1.f/256.f * acc1.s1) * sc8[0] +
)"
		R"(                                 (acc1.s2 + 1.f/256.f * acc1.s3) * sc8[1] * 1.f/16.f +
)"
		R"(                                 (acc2.s0 + 1.f/256.f * acc2.s1) * sc8[4] +
)"
		R"(                                 (acc2.s2 + 1.f/256.f * acc2.s3) * sc8[5] * 1.f/16.f) -
)"
		R"(                         dmin * (sumy.s0 * sc8[2] + sumy.s1 * sc8[3] + sumy.s2 * sc8[6] + sumy.s3 * sc8[7]);
)"
		R"(
)"
		R"(            q1 += nb01/2;
)"
		R"(            sc += nb01/2;
)"
		R"(            dh += nb01/2;
)"
		R"(        }
)"
		R"(
)"
		R"(        y4 += BLOCK_STRIDE * QK_K;
)"
		R"(    }
)"
		R"(
)"
		R"(    global float * dst_f32 = (global float *) dst + im*ne0*ne1 + r1*ne0;
)"
		R"(
)"
		R"(    for (int row = 0; row < N_DST; ++row) {
)"
		R"(        all_sum = sub_group_reduce_add(sumf[row]);
)"
		R"(        if (first_row + row < ne01) {
)"
		R"(            if (get_sub_group_local_id() == 0) {
)"
		R"(                dst_f32[first_row + row] = all_sum;
)"
		R"(            }
)"
		R"(        }
)"
		R"(    }
)"
		R"(}
)"
