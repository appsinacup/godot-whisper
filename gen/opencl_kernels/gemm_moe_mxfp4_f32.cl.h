R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
		R"(#pragma OPENCL EXTENSION cl_khr_subgroups : enable
)"
		R"(#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
)"
		R"(
)"
		R"(#define QK_MXFP4 32
)"
		R"(#define N_SIMDGROUP 2
)"
		R"(#define SIMDGROUP_WIDTH 64
)"
		R"(
)"
		R"(static inline half8 mxfp4_to_fp16_packed8(ushort2 fp4x8) { //, ushort 0x0E00, ushort 0x8000) {
)"
		R"(    ushort2 fp16_packed_a_0, fp16_packed_b_0, bias_a, bias_b, sign_a, sign_b;
)"
		R"(    fp16_packed_a_0.lo = (fp4x8.s0 << 9) & 0x0E00;
)"
		R"(    fp16_packed_a_0.hi = (fp4x8.s0 << 5) & 0x0E00;
)"
		R"(    fp16_packed_b_0.lo = (fp4x8.s0 << 1) & 0x0E00;
)"
		R"(    fp16_packed_b_0.hi = (fp4x8.s0 >> 3) & 0x0E00;
)"
		R"(
)"
		R"(    bias_a.lo = (fp16_packed_a_0.lo != 0) ? 0x3800 : 0x0;
)"
		R"(    bias_a.hi = (fp16_packed_a_0.hi != 0) ? 0x3800 : 0x0;
)"
		R"(    bias_b.lo = (fp16_packed_b_0.lo != 0) ? 0x3800 : 0x0;
)"
		R"(    bias_b.hi = (fp16_packed_b_0.hi != 0) ? 0x3800 : 0x0;
)"
		R"(
)"
		R"(    fp16_packed_a_0.lo = (fp16_packed_a_0.lo != 0x0200) ? fp16_packed_a_0.lo : 0x0;
)"
		R"(    fp16_packed_a_0.hi = (fp16_packed_a_0.hi != 0x0200) ? fp16_packed_a_0.hi : 0x0;
)"
		R"(    fp16_packed_b_0.lo = (fp16_packed_b_0.lo != 0x0200) ? fp16_packed_b_0.lo : 0x0;
)"
		R"(    fp16_packed_b_0.hi = (fp16_packed_b_0.hi != 0x0200) ? fp16_packed_b_0.hi : 0x0;
)"
		R"(
)"
		R"(    sign_a.lo = (fp4x8.s0 << 12) & 0x8000;
)"
		R"(    sign_a.hi = (fp4x8.s0 << 8) & 0x8000;
)"
		R"(    sign_b.lo = (fp4x8.s0 << 4) & 0x8000;
)"
		R"(    sign_b.hi = fp4x8.s0 & 0x8000;
)"
		R"(
)"
		R"(    fp16_packed_a_0 = sign_a + bias_a + fp16_packed_a_0;
)"
		R"(    fp16_packed_b_0 = sign_b + bias_b + fp16_packed_b_0;
)"
		R"(
)"
		R"(    ushort2 fp16_packed_a_1, fp16_packed_b_1;
)"
		R"(    fp16_packed_a_1.lo = (fp4x8.s1 << 9) & 0x0E00;
)"
		R"(    fp16_packed_a_1.hi = (fp4x8.s1 << 5) & 0x0E00;
)"
		R"(    fp16_packed_b_1.lo = (fp4x8.s1 << 1) & 0x0E00;
)"
		R"(    fp16_packed_b_1.hi = (fp4x8.s1 >> 3) & 0x0E00;
)"
		R"(
)"
		R"(    bias_a.lo = (fp16_packed_a_1.lo != 0) ? 0x3800 : 0x0;
)"
		R"(    bias_a.hi = (fp16_packed_a_1.hi != 0) ? 0x3800 : 0x0;
)"
		R"(    bias_b.lo = (fp16_packed_b_1.lo != 0) ? 0x3800 : 0x0;
)"
		R"(    bias_b.hi = (fp16_packed_b_1.hi != 0) ? 0x3800 : 0x0;
)"
		R"(
)"
		R"(    fp16_packed_a_1.lo = (fp16_packed_a_1.lo != 0x0200) ? fp16_packed_a_1.lo : 0x0;
)"
		R"(    fp16_packed_a_1.hi = (fp16_packed_a_1.hi != 0x0200) ? fp16_packed_a_1.hi : 0x0;
)"
		R"(    fp16_packed_b_1.lo = (fp16_packed_b_1.lo != 0x0200) ? fp16_packed_b_1.lo : 0x0;
)"
		R"(    fp16_packed_b_1.hi = (fp16_packed_b_1.hi != 0x0200) ? fp16_packed_b_1.hi : 0x0;
)"
		R"(
)"
		R"(    sign_a.lo = (fp4x8.s1 << 12) & 0x8000;
)"
		R"(    sign_a.hi = (fp4x8.s1 << 8) & 0x8000;
)"
		R"(    sign_b.lo = (fp4x8.s1 << 4) & 0x8000;
)"
		R"(    sign_b.hi = fp4x8.s1 & 0x8000;
)"
		R"(
)"
		R"(    fp16_packed_a_1 = sign_a + bias_a + fp16_packed_a_1;
)"
		R"(    fp16_packed_b_1 = sign_b + bias_b + fp16_packed_b_1;
)"
		R"(
)"
		R"(    return as_half8((ushort8)(fp16_packed_a_0, fp16_packed_b_0, fp16_packed_a_1, fp16_packed_b_1));
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
		R"(
)"
		R"(__attribute__((qcom_reqd_sub_group_size("half")))
)"
		R"(__kernel void kernel_gemm_moe_mxfp4_f32(
)"
		R"(    __global uint4 * src0_q,
)"
		R"(    __global uchar * src0_e,
)"
		R"(    __read_only image1d_buffer_t src1,
)"
		R"(    __global ushort4 * src2,
)"
		R"(    __global float * dst,
)"
		R"(    ulong         offsetd,
)"
		R"(    int           ne00,
)"
		R"(    int           ne01,
)"
		R"(    int           tile_size
)"
		R"() {
)"
		R"(    uint i01  = get_global_id(0);
)"
		R"(    uint i20  = get_global_id(2);
)"
		R"(    uint sgid = get_local_id(1);
)"
		R"(    uint slid = get_sub_group_local_id();
)"
		R"(
)"
		R"(    ushort4 router = src2[i20];
)"
		R"(    ushort expert_id = router.x;
)"
		R"(    ushort i11 = router.y;
)"
		R"(    ushort i1 = router.z;
)"
		R"(    ushort tile_id = router.w;
)"
		R"(
)"
		R"(    if (tile_id * tile_size + i01 >= ne01) { // handle edge case when ne01 is not multiple of tile_size
)"
		R"(        return;
)"
		R"(    }
)"
		R"(
)"
		R"(    uint expert_offset = expert_id * ne00 * ne01 / 32;
)"
		R"(    uint tile_offset = expert_offset + tile_id * tile_size + i01;
)"
		R"(
)"
		R"(    __private float sum = 0.0f; // each thread calculate partial sum of one output
)"
		R"(
)"
		R"(    // loop along ne00 in block granularity, skip 4 blocks every iter
)"
		R"(    for (uint ib00 = sgid; ib00 < (ne00 / QK_MXFP4); ib00 += N_SIMDGROUP) {
)"
		R"(        // load one block of q
)"
		R"(        uint4 regQ = src0_q[tile_offset + ib00 * ne01];
)"
		R"(        // convert 8 fp4 to fp16
)"
		R"(        half8 fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s0));
)"
		R"(
)"
		R"(        uint offset = i11 * ne00 / 4 + ib00 * 8;
)"
		R"(        float4 shared_y4;
)"
		R"(        shared_y4 = read_imagef(src1, (offset + 0));
)"
		R"(        float4 acc = shared_y4 * (float4)(fp16x8.s0, fp16x8.s2, fp16x8.s4, fp16x8.s6);
)"
		R"(
)"
		R"(        shared_y4 = read_imagef(src1, (offset + 4));
)"
		R"(        acc += shared_y4 * (float4)(fp16x8.s1, fp16x8.s3, fp16x8.s5, fp16x8.s7);
)"
		R"(
)"
		R"(
)"
		R"(        fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s1));
)"
		R"(
)"
		R"(        shared_y4 = read_imagef(src1, (offset + 1));
)"
		R"(        acc += shared_y4 * (float4)(fp16x8.s0, fp16x8.s2, fp16x8.s4, fp16x8.s6);
)"
		R"(
)"
		R"(        shared_y4 = read_imagef(src1, (offset + 5));
)"
		R"(        acc += shared_y4 * (float4)(fp16x8.s1, fp16x8.s3, fp16x8.s5, fp16x8.s7);
)"
		R"(
)"
		R"(
)"
		R"(        fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s2));
)"
		R"(
)"
		R"(        shared_y4 = read_imagef(src1, (offset + 2));
)"
		R"(        acc += shared_y4 * (float4)(fp16x8.s0, fp16x8.s2, fp16x8.s4, fp16x8.s6);
)"
		R"(
)"
		R"(        shared_y4 = read_imagef(src1, (offset + 6));
)"
		R"(        acc += shared_y4 * (float4)(fp16x8.s1, fp16x8.s3, fp16x8.s5, fp16x8.s7);
)"
		R"(
)"
		R"(
)"
		R"(        fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s3));
)"
		R"(
)"
		R"(        shared_y4 = read_imagef(src1, (offset + 3));
)"
		R"(        acc += shared_y4 * (float4)(fp16x8.s0, fp16x8.s2, fp16x8.s4, fp16x8.s6);
)"
		R"(
)"
		R"(        shared_y4 = read_imagef(src1, (offset + 7));
)"
		R"(        acc += shared_y4 * (float4)(fp16x8.s1, fp16x8.s3, fp16x8.s5, fp16x8.s7);
)"
		R"(
)"
		R"(        uchar regE = src0_e[tile_offset + ib00 * ne01];
)"
		R"(        sum += e8m0_to_fp32(regE) * ((acc.s0 + acc.s1) + (acc.s2 + acc.s3));
)"
		R"(    }
)"
		R"(
)"
		R"(    // reduction in local memory, assumes #subgroups=4
)"
		R"(    __local float reduceLM[SIMDGROUP_WIDTH * (N_SIMDGROUP - 1)];
)"
		R"(    if (sgid == 1) reduceLM[SIMDGROUP_WIDTH * 0 + slid] = sum;
)"
		R"(    // if (sgid == 2) reduceLM[SIMDGROUP_WIDTH * 1 + slid] = sum;
)"
		R"(    // if (sgid == 3) reduceLM[SIMDGROUP_WIDTH * 2 + slid] = sum;
)"
		R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
		R"(    if (sgid == 0) sum += reduceLM[SIMDGROUP_WIDTH * 0 + slid];
)"
		R"(    // if (sgid == 0) sum += reduceLM[SIMDGROUP_WIDTH * 1 + slid];
)"
		R"(    // if (sgid == 0) sum += reduceLM[SIMDGROUP_WIDTH * 2 + slid];
)"
		R"(
)"
		R"(    // 1 outputs per thread in subgroup 0
)"
		R"(    if (sgid == 0) {
)"
		R"(        dst = dst + (offsetd >> 2);
)"
		R"(        dst[i01 + tile_id * tile_size + i1 * ne01] = sum;
)"
		R"(    }
)"
		R"(
)"
		R"(}
)"
