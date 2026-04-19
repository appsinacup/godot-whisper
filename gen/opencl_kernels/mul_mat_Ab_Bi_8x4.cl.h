R"(// src0_q, src0_d, src1 are transposed as a preprocessing step
)"
R"(// 4-bit weights are transposed in groups of 4 (unsigned short int)
)"
R"(// consider weights originally "next to each other", now "on top of each other"
)"
R"(// each fiber computes a 8x4 tile of output elements
)"
R"(// using unshuffled weights
)"
R"(
)"
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
R"(kernel void kernel_mul_mat_Ab_Bi_8x4(
)"
R"(        global const ushort * src0_q,       // quantized A
)"
R"(        global const half  * src0_d,        // A scales
)"
R"(        __read_only image1d_buffer_t src1,  // B (1d image)
)"
R"(        global float * dst,                 // C
)"
R"(        int m,                              // M
)"
R"(        int n,                              // N with padding
)"
R"(        int k,                              // K
)"
R"(        int n_no_padding                    // N without padding
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
R"(    int gy = get_global_id(0);
)"
R"(    int gx = get_global_id(1);
)"
R"(    int gx_2 = gx << 2;
)"
R"(
)"
R"(    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0; // 8x4 output elements
)"
R"(    half8 B; // registers for activations
)"
R"(    half4 dequantized_weights; // registers for dequantized weights
)"
R"(    __global const ushort* weight_ptr = src0_q + gx_2; // pointer for weights
)"
R"(    __global const half* scale_ptr = src0_d + gx_2; // pointer for scales
)"
R"(
)"
R"(    for(int i=0; i<k; i+=4){ //loop through K dimension
)"
R"(
)"
R"(        B.s0123 = read_imageh(src1, gy*2 + (i)*(n_4));
)"
R"(        B.s4567 = read_imageh(src1, gy*2 + (i)*(n_4)+1);
)"
R"(
)"
R"(        // keep (i/4) and (i/32) in parenthesis, rounds down
)"
R"(        // load 4 consecutive groups of 4 weights
)"
R"(        ushort4 bits4 = vload4(0, weight_ptr + (i/4)*(m)); // (i/4) because weights grouped in 4s
)"
R"(
)"
R"(        // load 4 consecutive scales
)"
R"(        half4 scale = vload4(0, scale_ptr + (i/32)*(m));// (i/32) because 1 scale per 32 elements
)"
R"(
)"
R"(        // j=0
)"
R"(        dequantized_weights.s0 = ((bits4.s0 & (0x000F)) - 8) * scale.s0; // dequantize a row of the 16 weights
)"
R"(        dequantized_weights.s1 = ((bits4.s1 & (0x000F)) - 8) * scale.s1;
)"
R"(        dequantized_weights.s2 = ((bits4.s2 & (0x000F)) - 8) * scale.s2;
)"
R"(        dequantized_weights.s3 = ((bits4.s3 & (0x000F)) - 8) * scale.s3;
)"
R"(        c0 += B * dequantized_weights.s0; // vector-scalar multiplication to accumulate
)"
R"(        c1 += B * dequantized_weights.s1;
)"
R"(        c2 += B * dequantized_weights.s2;
)"
R"(        c3 += B * dequantized_weights.s3;
)"
R"(
)"
R"(        // j=1
)"
R"(        B.s0123 = read_imageh(src1, gy*2 + (i+1)*(n_4));
)"
R"(        B.s4567 = read_imageh(src1, gy*2 + (i+1)*(n_4)+1);
)"
R"(        dequantized_weights.s0 = (((bits4.s0 & (0x00F0)) >> 4) - 8) * scale.s0; // dequantize a row of the 16 weights
)"
R"(        dequantized_weights.s1 = (((bits4.s1 & (0x00F0)) >> 4) - 8) * scale.s1;
)"
R"(        dequantized_weights.s2 = (((bits4.s2 & (0x00F0)) >> 4) - 8) * scale.s2;
)"
R"(        dequantized_weights.s3 = (((bits4.s3 & (0x00F0)) >> 4) - 8) * scale.s3;
)"
R"(        c0 += B * dequantized_weights.s0; //vector-scalar multiplication to accumulate
)"
R"(        c1 += B * dequantized_weights.s1;
)"
R"(        c2 += B * dequantized_weights.s2;
)"
R"(        c3 += B * dequantized_weights.s3;
)"
R"(
)"
R"(        // j=2
)"
R"(        B.s0123 = read_imageh(src1, gy*2 + (i+2)*(n_4));
)"
R"(        B.s4567 = read_imageh(src1, gy*2 + (i+2)*(n_4)+1);
)"
R"(        dequantized_weights.s0 = (((bits4.s0 & (0x0F00)) >> 8) - 8) * scale.s0; // dequantize a row of the 16 weights
)"
R"(        dequantized_weights.s1 = (((bits4.s1 & (0x0F00)) >> 8) - 8) * scale.s1;
)"
R"(        dequantized_weights.s2 = (((bits4.s2 & (0x0F00)) >> 8) - 8) * scale.s2;
)"
R"(        dequantized_weights.s3 = (((bits4.s3 & (0x0F00)) >> 8) - 8) * scale.s3;
)"
R"(        c0 += B * dequantized_weights.s0; // vector-scalar multiplication to accumulate
)"
R"(        c1 += B * dequantized_weights.s1;
)"
R"(        c2 += B * dequantized_weights.s2;
)"
R"(        c3 += B * dequantized_weights.s3;
)"
R"(
)"
R"(        // j=3
)"
R"(        B.s0123 = read_imageh(src1, gy*2 + (i+3)*(n_4));
)"
R"(        B.s4567 = read_imageh(src1, gy*2 + (i+3)*(n_4)+1);
)"
R"(        dequantized_weights.s0 = (((bits4.s0 & (0xF000)) >> 12) - 8) * scale.s0; // dequantize a row of the 16 weights
)"
R"(        dequantized_weights.s1 = (((bits4.s1 & (0xF000)) >> 12) - 8) * scale.s1;
)"
R"(        dequantized_weights.s2 = (((bits4.s2 & (0xF000)) >> 12) - 8) * scale.s2;
)"
R"(        dequantized_weights.s3 = (((bits4.s3 & (0xF000)) >> 12) - 8) * scale.s3;
)"
R"(        c0 += B * dequantized_weights.s0; // vector-scalar multiplication to accumulate
)"
R"(        c1 += B * dequantized_weights.s1;
)"
R"(        c2 += B * dequantized_weights.s2;
)"
R"(        c3 += B * dequantized_weights.s3;
)"
R"(    }
)"
R"(
)"
R"(    int idx = (gy<<3)*m + (gx<<2); // vectorized store 16 elements
)"
R"(
)"
R"(    // conditional check if store is to a valid location. Required when N is not a multiple of 8
)"
R"(    // if statements allow registers to be reused for each store
)"
R"(    // provides a performance boost due to reduced register footprint, which increases number of concurrent waves
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
