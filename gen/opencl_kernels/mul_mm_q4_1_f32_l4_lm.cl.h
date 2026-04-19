R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
		R"(
)"
		R"(#define LOAD_VEC_A 8
)"
		R"(#define LOAD_VEC_B 4
)"
		R"(
)"
		R"(#define BM 64
)"
		R"(#define BN 64
)"
		R"(#define BK 32
)"
		R"(#define TM 4
)"
		R"(#define TN 8
)"
		R"(
)"
		R"(kernel void kernel_mul_mm_q4_1_f32_l4_lm(
)"
		R"(    global uchar4 * src0_q,
)"
		R"(    global half   * src0_d,
)"
		R"(    global half   * src0_m,
)"
		R"(    global float4 * src1,
)"
		R"(    ulong offset1,
)"
		R"(    global float  * dst,
)"
		R"(    ulong offsetd,
)"
		R"(
)"
		R"(    int ne00,
)"
		R"(    int ne01,
)"
		R"(    int ne02,
)"
		R"(    int ne11,
)"
		R"(    int ne12,
)"
		R"(
)"
		R"(    int stride_a,
)"
		R"(    int stride_b,
)"
		R"(    int stride_d,
)"
		R"(
)"
		R"(    int batch_stride_a,
)"
		R"(    int batch_stride_b,
)"
		R"(    int batch_stride_d,
)"
		R"(
)"
		R"(    int r2,
)"
		R"(    int r3
)"
		R"() {
)"
		R"(    src1 = (global float4*)((global char*)src1 + offset1);
)"
		R"(    dst  = (global float *)((global char*)dst  + offsetd);
)"
		R"(
)"
		R"(    local float buf_a[BM * BK];
)"
		R"(    local float buf_b[BN * BK];
)"
		R"(
)"
		R"(    const int batch_idx = get_global_id(2);
)"
		R"(
)"
		R"(    const int i13 = batch_idx / ne12;
)"
		R"(    const int i12 = batch_idx % ne12;
)"
		R"(
)"
		R"(    const int i03 = i13 / r3;
)"
		R"(    const int i02 = i12 / r2;
)"
		R"(
)"
		R"(    const int batch_idx_a = i03 * ne02 + i02;
)"
		R"(
)"
		R"(    const int ir = get_group_id(0);
)"
		R"(    const int ic = get_group_id(1);
)"
		R"(
)"
		R"(    const int tid = get_local_id(0);
)"
		R"(    const int th_r  = tid % (BM / TM);
)"
		R"(    const int th_c  = tid / (BM / TM);
)"
		R"(
)"
		R"(    const int loadr_a = get_local_id(0) % (BK / LOAD_VEC_A);
)"
		R"(    const int loadc_a = get_local_id(0) / (BK / LOAD_VEC_A);
)"
		R"(    const int loadr_b = get_local_id(0) % (BK / LOAD_VEC_B);
)"
		R"(    const int loadc_b = get_local_id(0) / (BK / LOAD_VEC_B);
)"
		R"(
)"
		R"(    const int loadstride_a = get_local_size(0) * LOAD_VEC_A / BK;
)"
		R"(    const int loadstride_b = get_local_size(0) * LOAD_VEC_B / BK;
)"
		R"(
)"
		R"(    int pos_a = (batch_idx_a * batch_stride_a + ir * BM * stride_a) / LOAD_VEC_A;
)"
		R"(    int pos_b = (batch_idx   * batch_stride_b + ic * BN * stride_b) / LOAD_VEC_B;
)"
		R"(
)"
		R"(    float sums[TM * TN];
)"
		R"(    float cache_a[TM];
)"
		R"(    float cache_b[TN];
)"
		R"(
)"
		R"(    for (int i = 0; i < TM * TN; i++) {
)"
		R"(        sums[i] = 0.0f;
)"
		R"(    }
)"
		R"(
)"
		R"(    for (int block = 0; block < ne00; block += BK) {
)"
		R"(        for (int l = 0; l < BM; l += loadstride_a) {
)"
		R"(            if (ir*BM + loadc_a + l < ne01) {
)"
		R"(                int idx = pos_a + (loadc_a + l) * stride_a / LOAD_VEC_A + loadr_a;
)"
		R"(                int ib  = idx / 4;
)"
		R"(                int iqs = idx % 4;
)"
		R"(
)"
		R"(                float d = (float)src0_d[ib];
)"
		R"(                float m = (float)src0_m[ib];
)"
		R"(                global uchar4 * qs = src0_q + ib*4 + iqs;
)"
		R"(                uchar4 q = *qs;
)"
		R"(                float4 v1 = (convert_float4((uchar4)((q.s0   )&0x0F, (q.s1   )&0x0F, (q.s2   )&0x0F, (q.s3   )&0x0F)))*d + m;
)"
		R"(                float4 v2 = (convert_float4((uchar4)((q.s0>>4)&0x0F, (q.s1>>4)&0x0F, (q.s2>>4)&0x0F, (q.s3>>4)&0x0F)))*d + m;
)"
		R"(
)"
		R"(                buf_a[(loadr_a * 4 +  0) * BM + loadc_a + l] = v1.s0;
)"
		R"(                buf_a[(loadr_a * 4 +  1) * BM + loadc_a + l] = v1.s1;
)"
		R"(                buf_a[(loadr_a * 4 +  2) * BM + loadc_a + l] = v1.s2;
)"
		R"(                buf_a[(loadr_a * 4 +  3) * BM + loadc_a + l] = v1.s3;
)"
		R"(                buf_a[(loadr_a * 4 + 16) * BM + loadc_a + l] = v2.s0;
)"
		R"(                buf_a[(loadr_a * 4 + 17) * BM + loadc_a + l] = v2.s1;
)"
		R"(                buf_a[(loadr_a * 4 + 18) * BM + loadc_a + l] = v2.s2;
)"
		R"(                buf_a[(loadr_a * 4 + 19) * BM + loadc_a + l] = v2.s3;
)"
		R"(            } else {
)"
		R"(                buf_a[(loadr_a * 4 +  0) * BM + loadc_a + l] = 0.0f;
)"
		R"(                buf_a[(loadr_a * 4 +  1) * BM + loadc_a + l] = 0.0f;
)"
		R"(                buf_a[(loadr_a * 4 +  2) * BM + loadc_a + l] = 0.0f;
)"
		R"(                buf_a[(loadr_a * 4 +  3) * BM + loadc_a + l] = 0.0f;
)"
		R"(                buf_a[(loadr_a * 4 + 16) * BM + loadc_a + l] = 0.0f;
)"
		R"(                buf_a[(loadr_a * 4 + 17) * BM + loadc_a + l] = 0.0f;
)"
		R"(                buf_a[(loadr_a * 4 + 18) * BM + loadc_a + l] = 0.0f;
)"
		R"(                buf_a[(loadr_a * 4 + 19) * BM + loadc_a + l] = 0.0f;
)"
		R"(            }
)"
		R"(        }
)"
		R"(
)"
		R"(        for (int l = 0; l < BN; l += loadstride_b) {
)"
		R"(            if (ic*BN + loadc_b + l < ne11) {
)"
		R"(                int idx = pos_b + (loadc_b + l) * stride_b / LOAD_VEC_B + loadr_b;
)"
		R"(                buf_b[(loadr_b * LOAD_VEC_B + 0) * BN + loadc_b + l] = src1[idx].s0;
)"
		R"(                buf_b[(loadr_b * LOAD_VEC_B + 1) * BN + loadc_b + l] = src1[idx].s1;
)"
		R"(                buf_b[(loadr_b * LOAD_VEC_B + 2) * BN + loadc_b + l] = src1[idx].s2;
)"
		R"(                buf_b[(loadr_b * LOAD_VEC_B + 3) * BN + loadc_b + l] = src1[idx].s3;
)"
		R"(            } else {
)"
		R"(                buf_b[(loadr_b * LOAD_VEC_B + 0) * BN + loadc_b + l] = 0.0f;
)"
		R"(                buf_b[(loadr_b * LOAD_VEC_B + 1) * BN + loadc_b + l] = 0.0f;
)"
		R"(                buf_b[(loadr_b * LOAD_VEC_B + 2) * BN + loadc_b + l] = 0.0f;
)"
		R"(                buf_b[(loadr_b * LOAD_VEC_B + 3) * BN + loadc_b + l] = 0.0f;
)"
		R"(            }
)"
		R"(        }
)"
		R"(
)"
		R"(        barrier(CLK_LOCAL_MEM_FENCE);
)"
		R"(
)"
		R"(        pos_a += BK / LOAD_VEC_A;
)"
		R"(        pos_b += BK / LOAD_VEC_B;
)"
		R"(
)"
		R"(        for (int i = 0; i < BK; i++) {
)"
		R"(            for (int j = 0; j < TM; j++) {
)"
		R"(                cache_a[j] = buf_a[(i) * BM + th_r * TM + j];
)"
		R"(            }
)"
		R"(
)"
		R"(            for (int j = 0; j < TN; j++) {
)"
		R"(                cache_b[j] = buf_b[(i) * BN + th_c * TN + j];
)"
		R"(            }
)"
		R"(
)"
		R"(            for (int cc = 0; cc < TN; cc++) {
)"
		R"(                for (int cr = 0; cr < TM; cr++) {
)"
		R"(                    const int sums_idx = cc*TM + cr;
)"
		R"(                    sums[sums_idx] = mad(cache_a[cr], cache_b[cc], sums[sums_idx]);
)"
		R"(                }
)"
		R"(            }
)"
		R"(        }
)"
		R"(        barrier(CLK_LOCAL_MEM_FENCE);
)"
		R"(    }
)"
		R"(
)"
		R"(    const int dr = ir * BM + th_r * TM;
)"
		R"(    const int dc = ic * BN + th_c * TN;
)"
		R"(
)"
		R"(    const int offsets = batch_idx * batch_stride_d;
)"
		R"(
)"
		R"(    for (int cc = 0; cc < TN; cc++) {
)"
		R"(        for (int cr = 0; cr < TM; cr++) {
)"
		R"(            if (dr + cr < ne01 && dc + cc < ne11) {
)"
		R"(                dst[offsets + (dc + cc) * stride_d + dr + cr] = sums[cc * TM + cr];
)"
		R"(            }
)"
		R"(        }
)"
		R"(    }
)"
		R"(}
)"
