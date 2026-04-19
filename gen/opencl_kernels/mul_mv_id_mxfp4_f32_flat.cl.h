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
		R"(#define QK_MXFP4 32
)"
		R"(
)"
		R"(static inline half4 mxfp4_to_fp16_packed(ushort fp4x4) {
)"
		R"(    ushort2 fp16_packed_a, fp16_packed_b, bias_a, bias_b, sign_a, sign_b;
)"
		R"(    fp16_packed_a.lo = (fp4x4 << 9) & 0x0E00;
)"
		R"(    fp16_packed_a.hi = (fp4x4 << 5) & 0x0E00;
)"
		R"(    fp16_packed_b.lo = (fp4x4 << 1) & 0x0E00;
)"
		R"(    fp16_packed_b.hi = (fp4x4 >> 3) & 0x0E00;
)"
		R"(
)"
		R"(    bias_a.lo = (fp16_packed_a.lo == 0) ? 0x0 : 0x3800;
)"
		R"(    bias_a.hi = (fp16_packed_a.hi == 0) ? 0x0 : 0x3800;
)"
		R"(    bias_b.lo = (fp16_packed_b.lo == 0) ? 0x0 : 0x3800;
)"
		R"(    bias_b.hi = (fp16_packed_b.hi == 0) ? 0x0 : 0x3800;
)"
		R"(
)"
		R"(    fp16_packed_a.lo = (fp16_packed_a.lo == 0x0200) ? 0x0 : fp16_packed_a.lo;
)"
		R"(    fp16_packed_a.hi = (fp16_packed_a.hi == 0x0200) ? 0x0 : fp16_packed_a.hi;
)"
		R"(    fp16_packed_b.lo = (fp16_packed_b.lo == 0x0200) ? 0x0 : fp16_packed_b.lo;
)"
		R"(    fp16_packed_b.hi = (fp16_packed_b.hi == 0x0200) ? 0x0 : fp16_packed_b.hi;
)"
		R"(
)"
		R"(    sign_a.lo = (fp4x4 << 12) & 0x8000;
)"
		R"(    sign_a.hi = (fp4x4 << 8) & 0x8000;
)"
		R"(    sign_b.lo = (fp4x4 << 4) & 0x8000;
)"
		R"(    sign_b.hi = fp4x4 & 0x8000;
)"
		R"(
)"
		R"(    fp16_packed_a = sign_a + bias_a + fp16_packed_a;
)"
		R"(    fp16_packed_b = sign_b + bias_b + fp16_packed_b;
)"
		R"(
)"
		R"(    return as_half4((ushort4)(fp16_packed_a, fp16_packed_b));
)"
		R"(}
)"
		R"(
)"
		R"(static inline float e8m0_to_fp32(uchar x) {
)"
		R"(    int bits;
)"
		R"(    bits = (x == 0) ? 0x00400000 : ((uint) x << 23);
)"
		R"(    return as_float(bits);
)"
		R"(}
)"
		R"(
)"
		R"(#ifdef INTEL_GPU
)"
		R"(#define N_R0_MXFP4 2 // number of rows each subgroup works on
)"
		R"(#define N_SG_MXFP4 2 // number of subgroups in a work group
)"
		R"(#define N_SIMDWIDTH 16 // subgroup size
)"
		R"(#elif defined (ADRENO_GPU)
)"
		R"(#define N_R0_MXFP4 4
)"
		R"(#define N_SG_MXFP4 1
)"
		R"(#define N_SIMDWIDTH 64
)"
		R"(#define SRC0Q_IMG
)"
		R"(#endif
)"
		R"(
)"
		R"(kernel void kernel_mul_mv_id_mxfp4_f32_flat(
)"
		R"(#ifdef SRC0Q_IMG
)"
		R"(    __read_only image1d_buffer_t src0_q,
)"
		R"(#else
)"
		R"(    global uchar * src0_q,
)"
		R"(#endif
)"
		R"(    global uchar * src0_e,
)"
		R"(    global uchar * src1,
)"
		R"(    ulong         offset1,
)"
		R"(    global uchar * src2,
)"
		R"(    ulong         offset2,
)"
		R"(    global uchar * dst,
)"
		R"(    ulong         offsetd,
)"
		R"(    int           ne00,
)"
		R"(    ulong         nb01,
)"
		R"(    ulong         nb02,
)"
		R"(    ulong         nb03,
)"
		R"(    int           ne11,
)"
		R"(    int           ne12,
)"
		R"(    ulong         nb11,
)"
		R"(    ulong         nb12,
)"
		R"(    ulong         nb13,
)"
		R"(    int           ne20,
)"
		R"(    int           ne21,
)"
		R"(    ulong         nb21,
)"
		R"(    int           ne0,
)"
		R"(    int           ne1,
)"
		R"(    int           r2,
)"
		R"(    int           r3
)"
		R"() {
)"
		R"(    dst  = dst  + offsetd;
)"
		R"(
)"
		R"(    const int iid1 = get_group_id(2) / ne20;
)"
		R"(    const int idx  = get_group_id(2) % ne20;
)"
		R"(
)"
		R"(    uint i02 = ((global uint *) (src2 + offset2 + iid1 * nb21))[idx];
)"
		R"(
)"
		R"(    int i11 = idx % ne11;
)"
		R"(
)"
		R"(    int nb = ne00 / QK_MXFP4;
)"
		R"(
)"
		R"(    uint src0_off = i02*nb02;
)"
		R"(    src0_off /= 17; // 17 = sizeof(block_mxfp4)
)"
		R"(
)"
		R"(    src0_e = src0_e + src0_off;
)"
		R"(
)"
		R"(    dst = dst + (idx * ne0 + iid1 * ne1 * ne0) * sizeof(float);
)"
		R"(
)"
		R"(    int r0 = get_group_id(0);
)"
		R"(    int r1 = get_group_id(1);
)"
		R"(
)"
		R"(    int first_row = (r0 * N_SG_MXFP4 + get_sub_group_id()) * N_R0_MXFP4;
)"
		R"(
)"
		R"(    uint offset_src0 = first_row*nb01;
)"
		R"(    offset_src0 /= 17; // 17 = sizeof(block_mxfp4)
)"
		R"(#ifdef SRC0Q_IMG
)"
		R"(    ulong offset_q = src0_off + offset_src0;
)"
		R"(#else
)"
		R"(    src0_q = src0_q + src0_off*16;
)"
		R"(    global uchar16 * x_q = (global uchar16 *)(src0_q) + offset_src0;
)"
		R"(#endif
)"
		R"(    global uchar * x_e = src0_e + offset_src0;
)"
		R"(
)"
		R"(    const short ix = get_sub_group_local_id() >> 1;
)"
		R"(    const short it = get_sub_group_local_id() & 1;
)"
		R"(
)"
		R"(    float sumf[N_R0_MXFP4] = {0.f};
)"
		R"(
)"
		R"(    src1 = src1 + offset1 + i11 * nb11 + iid1 * nb12;
)"
		R"(    global float * y   = (global float *) (src1 + r1 * nb11);
)"
		R"(    global float * yb = y + ix * QK_MXFP4 + it * 8;
)"
		R"(
)"
		R"(    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH / 2) {
)"
		R"(        global float4 * y4 = (global float4 *)yb;
)"
		R"(
)"
		R"(        #pragma unroll
)"
		R"(        for (short row = 0; row < N_R0_MXFP4; row++) {
)"
		R"(            uchar xb_e = x_e[row * nb + ib];
)"
		R"(#ifdef SRC0Q_IMG
)"
		R"(            ushort4 xb_q = as_ushort4(read_imageui(src0_q, (offset_q + row * nb + ib) * 2 + it).xy);
)"
		R"(#else
)"
		R"(            ushort4 xb_q = vload4(0, (global ushort *)((global uchar *)(x_q + row * nb + ib) + 8 * it));
)"
		R"(#endif
)"
		R"(
)"
		R"(            half4 fp16x4_0 = mxfp4_to_fp16_packed(xb_q.s0);
)"
		R"(            half4 fp16x4_1 = mxfp4_to_fp16_packed(xb_q.s1);
)"
		R"(            float4 acc1 = y4[0] * (float4)(fp16x4_0.s0, fp16x4_0.s2, fp16x4_1.s0, fp16x4_1.s2);
)"
		R"(            acc1 += y4[4] * (float4)(fp16x4_0.s1, fp16x4_0.s3, fp16x4_1.s1, fp16x4_1.s3);
)"
		R"(
)"
		R"(            fp16x4_0 = mxfp4_to_fp16_packed(xb_q.s2);
)"
		R"(            fp16x4_1 = mxfp4_to_fp16_packed(xb_q.s3);
)"
		R"(            acc1 += y4[1] * (float4)(fp16x4_0.s0, fp16x4_0.s2, fp16x4_1.s0, fp16x4_1.s2);
)"
		R"(            acc1 += y4[5] * (float4)(fp16x4_0.s1, fp16x4_0.s3, fp16x4_1.s1, fp16x4_1.s3);
)"
		R"(
)"
		R"(            sumf[row] += e8m0_to_fp32(xb_e) * ((acc1.s0 + acc1.s1) + (acc1.s2 + acc1.s3));
)"
		R"(        }
)"
		R"(
)"
		R"(        yb += (N_SIMDWIDTH / 2) * QK_MXFP4;
)"
		R"(    }
)"
		R"(
)"
		R"(    global float * dst_f32 = (global float *)dst + (ulong)r1 * ne0;
)"
		R"(
)"
		R"(    for (int row = 0; row < N_R0_MXFP4 && first_row + row < ne0; ++row) {
)"
		R"(        float sum_all = sub_group_reduce_add(sumf[row]);
)"
		R"(        if (get_sub_group_local_id() == 0) {
)"
		R"(            dst_f32[first_row + row] = sum_all;
)"
		R"(        }
)"
		R"(    }
)"
		R"(}
)"
