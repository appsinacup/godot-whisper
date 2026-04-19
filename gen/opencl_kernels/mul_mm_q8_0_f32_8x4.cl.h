R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
)"
R"(
)"
R"(#ifdef cl_qcom_reqd_sub_group_size
)"
R"(#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
)"
R"(#define ADRENO_GPU 1
)"
R"(#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
)"
R"(#endif
)"
R"(
)"
R"(#ifdef ADRENO_GPU
)"
R"(REQD_SUBGROUP_SIZE_128
)"
R"(#endif
)"
R"(
)"
R"(kernel void kernel_mul_mm_q8_0_f32_8x4(
)"
R"(        global const uint * src0_q,
)"
R"(        global const half  * src0_d,
)"
R"(        __read_only image1d_buffer_t src1,
)"
R"(        global float * dst,
)"
R"(        int k,
)"
R"(        int m,
)"
R"(        int n,
)"
R"(        int n_no_padding,
)"
R"(        ulong offsetd
)"
R"() {
)"
R"(
)"
R"(    int m_4 = m >> 2;
)"
R"(    int n_4 = n >> 2;
)"
R"(
)"
R"(    int gy   = get_global_id(0);
)"
R"(    int gx   = get_global_id(1);
)"
R"(    int gx_2 = gx << 2;
)"
R"(    dst  = (global float *)((global char*)dst  + offsetd);
)"
R"(
)"
R"(
)"
R"(    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
)"
R"(    half8 B;
)"
R"(    half4 deq;
)"
R"(
)"
R"(    __global const uint* wptr = src0_q + gx_2;
)"
R"(    __global const half* sptr = src0_d + gx_2;
)"
R"(
)"
R"(      for (int i = 0; i < k; i += 4) {
)"
R"(        uint4 pack4 = vload4(0, wptr + (i / 4) * m);
)"
R"(        half4 scale = vload4(0, sptr + (i / 32) * m);
)"
R"(
)"
R"(        char4 p0 = as_char4(pack4.s0);
)"
R"(        char4 p1 = as_char4(pack4.s1);
)"
R"(        char4 p2 = as_char4(pack4.s2);
)"
R"(        char4 p3 = as_char4(pack4.s3);
)"
R"(
)"
R"(        // ------------------- j = 0 (k = i+0) -------------------
)"
R"(        B.s0123 = read_imageh(src1, gy * 2 + (i + 0) * n_4);
)"
R"(        B.s4567 = read_imageh(src1, gy * 2 + (i + 0) * n_4 + 1);
)"
R"(
)"
R"(        half4 wj0 = convert_half4((char4)(p0.s0, p1.s0, p2.s0, p3.s0)) * scale;
)"
R"(
)"
R"(        c0 += B * wj0.s0;
)"
R"(        c1 += B * wj0.s1;
)"
R"(        c2 += B * wj0.s2;
)"
R"(        c3 += B * wj0.s3;
)"
R"(
)"
R"(        // ------------------- j = 1 (k = i+1) -------------------
)"
R"(        B.s0123 = read_imageh(src1, gy * 2 + (i + 1) * n_4);
)"
R"(        B.s4567 = read_imageh(src1, gy * 2 + (i + 1) * n_4 + 1);
)"
R"(
)"
R"(        half4 wj1 = convert_half4((char4)(p0.s1, p1.s1, p2.s1, p3.s1)) * scale;
)"
R"(
)"
R"(        c0 += B * wj1.s0;
)"
R"(        c1 += B * wj1.s1;
)"
R"(        c2 += B * wj1.s2;
)"
R"(        c3 += B * wj1.s3;
)"
R"(
)"
R"(        // ------------------- j = 2 (k = i+2) -------------------
)"
R"(        B.s0123 = read_imageh(src1, gy * 2 + (i + 2) * n_4);
)"
R"(        B.s4567 = read_imageh(src1, gy * 2 + (i + 2) * n_4 + 1);
)"
R"(
)"
R"(        half4 wj2 = convert_half4((char4)(p0.s2, p1.s2, p2.s2, p3.s2)) * scale;
)"
R"(
)"
R"(        c0 += B * wj2.s0;
)"
R"(        c1 += B * wj2.s1;
)"
R"(        c2 += B * wj2.s2;
)"
R"(        c3 += B * wj2.s3;
)"
R"(
)"
R"(        // ------------------- j = 3 (k = i+3) -------------------
)"
R"(        B.s0123 = read_imageh(src1, gy * 2 + (i + 3) * n_4);
)"
R"(        B.s4567 = read_imageh(src1, gy * 2 + (i + 3) * n_4 + 1);
)"
R"(
)"
R"(        half4 wj3 = convert_half4((char4)(p0.s3, p1.s3, p2.s3, p3.s3)) * scale;
)"
R"(
)"
R"(        c0 += B * wj3.s0;
)"
R"(        c1 += B * wj3.s1;
)"
R"(        c2 += B * wj3.s2;
)"
R"(        c3 += B * wj3.s3;
)"
R"(    }
)"
R"(
)"
R"(    int idx = (gy << 3) * m + (gx << 2);
)"
R"(
)"
R"(    if(idx+3 < m*n_no_padding){
)"
R"(        vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst + idx);
)"
R"(        idx += m;
)"
R"(    }
)"
R"(    if(idx+3 < m*n_no_padding){
)"
R"(        vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst + idx);
)"
R"(        idx += m;
)"
R"(    }
)"
R"(    if(idx+3 < m*n_no_padding){
)"
R"(        vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst + idx);
)"
R"(        idx += m;
)"
R"(    }
)"
R"(    if(idx+3 < m*n_no_padding){
)"
R"(        vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst + idx);
)"
R"(        idx += m;
)"
R"(    }
)"
R"(    if(idx+3 < m*n_no_padding){
)"
R"(        vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst + idx);
)"
R"(        idx += m;
)"
R"(    }
)"
R"(    if(idx+3 < m*n_no_padding){
)"
R"(        vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst + idx);
)"
R"(        idx += m;
)"
R"(    }
)"
R"(    if(idx+3 < m*n_no_padding){
)"
R"(        vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst + idx);
)"
R"(        idx += m;
)"
R"(    }
)"
R"(    if(idx+3 < m*n_no_padding){
)"
R"(        vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst + idx);
)"
R"(    }
)"
R"(}
)"
