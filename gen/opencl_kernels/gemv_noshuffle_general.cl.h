R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(#pragma OPENCL EXTENSION cl_khr_subgroups : enable
)"
R"(
)"
R"(#ifdef cl_qcom_reqd_sub_group_size
)"
R"(#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
)"
R"(#define ADRENO_GPU 1
)"
R"(#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
)"
R"(#endif
)"
R"(
)"
R"(// assume
)"
R"(#define QK4_0 32
)"
R"(#define N_SIMDGROUP 4
)"
R"(
)"
R"(#define dequantizeBlockAccum_ns_sgbroadcast_1_hi(total_sums, bits4, scale, y) \
)"
R"(    float shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s0, 0); \
)"
R"(    total_sums.s0 += ((bits4.s0 & 0x000F) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += ((bits4.s1 & 0x000F) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s1, 0); \
)"
R"(    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s2, 0); \
)"
R"(    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s3, 0); \
)"
R"(    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s4, 0); \
)"
R"(    total_sums.s0 += ((bits4.s2 & 0x000F) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += ((bits4.s3 & 0x000F) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s5, 0); \
)"
R"(    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s6, 0); \
)"
R"(    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s7, 0); \
)"
R"(    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s0, 1); \
)"
R"(    total_sums.s0 += ((bits4.s4 & 0x000F) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += ((bits4.s5 & 0x000F) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s1, 1); \
)"
R"(    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s2, 1); \
)"
R"(    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s3, 1); \
)"
R"(    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s4, 1); \
)"
R"(    total_sums.s0 += ((bits4.s6 & 0x000F) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += ((bits4.s7 & 0x000F) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s5, 1); \
)"
R"(    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s6, 1); \
)"
R"(    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s7, 1); \
)"
R"(    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
)"
R"(
)"
R"(
)"
R"(#define dequantizeBlockAccum_ns_sgbroadcast_1_lo(total_sums, bits4, scale, y) \
)"
R"(    shared_y = sub_group_broadcast(y.s0, 2); \
)"
R"(    total_sums.s0 += ((bits4.s0 & 0x000F) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += ((bits4.s1 & 0x000F) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s1, 2); \
)"
R"(    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s2, 2); \
)"
R"(    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s3, 2); \
)"
R"(    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s4, 2); \
)"
R"(    total_sums.s0 += ((bits4.s2 & 0x000F) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += ((bits4.s3 & 0x000F) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s5, 2); \
)"
R"(    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s6, 2); \
)"
R"(    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s7, 2); \
)"
R"(    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s0, 3); \
)"
R"(    total_sums.s0 += ((bits4.s4 & 0x000F) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += ((bits4.s5 & 0x000F) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s1, 3); \
)"
R"(    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s2, 3); \
)"
R"(    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s3, 3); \
)"
R"(    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s4, 3); \
)"
R"(    total_sums.s0 += ((bits4.s6 & 0x000F) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += ((bits4.s7 & 0x000F) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s5, 3); \
)"
R"(    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s6, 3); \
)"
R"(    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y.s7, 3); \
)"
R"(    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
)"
R"(    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
)"
R"(
)"
R"(
)"
R"(#define dequantizeBlockAccum_ns_sgbroadcast_8_hi(total_sums, bits4, scale, y) \
)"
R"(    float8 shared_y; \
)"
R"(    shared_y = sub_group_broadcast(y, 0); \
)"
R"(    total_sums.s0 += ((bits4.s0 & 0x000F) - 8) * scale.s0 * shared_y.s0; \
)"
R"(    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s1; \
)"
R"(    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s2; \
)"
R"(    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s3; \
)"
R"(    total_sums.s0 += ((bits4.s2 & 0x000F) - 8) * scale.s0 * shared_y.s4; \
)"
R"(    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s5; \
)"
R"(    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s6; \
)"
R"(    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s7; \
)"
R"(    total_sums.s1 += ((bits4.s1 & 0x000F) - 8) * scale.s1 * shared_y.s0; \
)"
R"(    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s1; \
)"
R"(    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s2; \
)"
R"(    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s3; \
)"
R"(    total_sums.s1 += ((bits4.s3 & 0x000F) - 8) * scale.s1 * shared_y.s4; \
)"
R"(    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s5; \
)"
R"(    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s6; \
)"
R"(    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s7; \
)"
R"(    shared_y = sub_group_broadcast(y, 1); \
)"
R"(    total_sums.s0 += ((bits4.s4 & 0x000F) - 8) * scale.s0 * shared_y.s0; \
)"
R"(    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s1; \
)"
R"(    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s2; \
)"
R"(    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s3; \
)"
R"(    total_sums.s0 += ((bits4.s6 & 0x000F) - 8) * scale.s0 * shared_y.s4; \
)"
R"(    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s5; \
)"
R"(    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s6; \
)"
R"(    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s7; \
)"
R"(    total_sums.s1 += ((bits4.s5 & 0x000F) - 8) * scale.s1 * shared_y.s0; \
)"
R"(    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s1; \
)"
R"(    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s2; \
)"
R"(    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s3; \
)"
R"(    total_sums.s1 += ((bits4.s7 & 0x000F) - 8) * scale.s1 * shared_y.s4; \
)"
R"(    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s5; \
)"
R"(    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s6; \
)"
R"(    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s7; \
)"
R"(
)"
R"(
)"
R"(#define dequantizeBlockAccum_ns_sgbroadcast_8_lo(total_sums, bits4, scale, y) \
)"
R"(    shared_y = sub_group_broadcast(y, 2); \
)"
R"(    total_sums.s0 += ((bits4.s0 & 0x000F) - 8) * scale.s0 * shared_y.s0; \
)"
R"(    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s1; \
)"
R"(    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s2; \
)"
R"(    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s3; \
)"
R"(    total_sums.s0 += ((bits4.s2 & 0x000F) - 8) * scale.s0 * shared_y.s4; \
)"
R"(    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s5; \
)"
R"(    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s6; \
)"
R"(    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s7; \
)"
R"(    total_sums.s1 += ((bits4.s1 & 0x000F) - 8) * scale.s1 * shared_y.s0; \
)"
R"(    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s1; \
)"
R"(    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s2; \
)"
R"(    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s3; \
)"
R"(    total_sums.s1 += ((bits4.s3 & 0x000F) - 8) * scale.s1 * shared_y.s4; \
)"
R"(    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s5; \
)"
R"(    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s6; \
)"
R"(    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s7; \
)"
R"(    shared_y = sub_group_broadcast(y, 3); \
)"
R"(    total_sums.s0 += ((bits4.s4 & 0x000F) - 8) * scale.s0 * shared_y.s0; \
)"
R"(    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s1; \
)"
R"(    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s2; \
)"
R"(    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s3; \
)"
R"(    total_sums.s0 += ((bits4.s6 & 0x000F) - 8) * scale.s0 * shared_y.s4; \
)"
R"(    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y.s5; \
)"
R"(    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y.s6; \
)"
R"(    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) - 8) * scale.s0 * shared_y.s7; \
)"
R"(    total_sums.s1 += ((bits4.s5 & 0x000F) - 8) * scale.s1 * shared_y.s0; \
)"
R"(    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s1; \
)"
R"(    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s2; \
)"
R"(    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s3; \
)"
R"(    total_sums.s1 += ((bits4.s7 & 0x000F) - 8) * scale.s1 * shared_y.s4; \
)"
R"(    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y.s5; \
)"
R"(    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y.s6; \
)"
R"(    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) - 8) * scale.s1 * shared_y.s7; \
)"
R"(
)"
R"(#ifdef ADRENO_GPU
)"
R"(REQD_SUBGROUP_SIZE_64
)"
R"(#endif
)"
R"(__kernel void kernel_gemv_noshuffle(
)"
R"(        __read_only  image1d_buffer_t src0_q,  // quantized A
)"
R"(        global half2  * src0_d,  // A scales
)"
R"(        __read_only  image1d_buffer_t src1,    // B
)"
R"(        ulong offset1,            // offset to B (0)
)"
R"(        global float * dst,     // C
)"
R"(        ulong offsetd,            // offset to C (0)
)"
R"(        int ne00,               // K
)"
R"(        int ne01,               // M
)"
R"(        int ne02,               // 1
)"
R"(        int ne10,               // K
)"
R"(        int ne12,               // 1
)"
R"(        int ne0,                // M
)"
R"(        int ne1,                // N
)"
R"(        int r2,                 // 1
)"
R"(        int r3)
)"
R"({
)"
R"(    uint groupId = get_local_id(1);
)"
R"(    uint gid     = get_global_id(0);
)"
R"(    ushort slid    = get_sub_group_local_id();
)"
R"(
)"
R"(    uint K = ne00;
)"
R"(    uint M = ne01;
)"
R"(
)"
R"(    uint LINE_STRIDE_A = M / 2;
)"
R"(    uint BLOCK_STRIDE_A = N_SIMDGROUP * M;
)"
R"(
)"
R"(    __private uint4     regA;
)"
R"(    __private half2     regS;
)"
R"(    __private float8    regB;
)"
R"(
)"
R"(    __private float2 totalSum = (float2)(0.0f);
)"
R"(
)"
R"(    // loop along K in block granularity, skip 4 blocks every iter
)"
R"(    for (uint k = groupId; k < (K / QK4_0); k += N_SIMDGROUP) {
)"
R"(        regS = src0_d[gid + k * LINE_STRIDE_A]; // each fiber loads scale of two rows
)"
R"(        // first 4 fibers in each wave load 8 B values to its private scope
)"
R"(        if (slid < 4) {
)"
R"(            regB.s0123 = read_imagef(src1, (slid * 2 + k * 8));
)"
R"(            regB.s4567 = read_imagef(src1, (1 + slid * 2 + k * 8));
)"
R"(        }
)"
R"(
)"
R"(        // load half weights for two blocks in consecutive rows
)"
R"(        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
)"
R"(        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
)"
R"(        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
)"
R"(        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
)"
R"(#ifdef VECTOR_SUB_GROUP_BROADCAT
)"
R"(        dequantizeBlockAccum_ns_sgbroadcast_8_hi(totalSum, as_ushort8(regA), regS, regB);
)"
R"(#else
)"
R"(        dequantizeBlockAccum_ns_sgbroadcast_1_hi(totalSum, as_ushort8(regA), regS, regB);
)"
R"(#endif // VECTOR_SUB_GROUP_BROADCAT
)"
R"(
)"
R"(        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
)"
R"(        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
)"
R"(        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
)"
R"(        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;
)"
R"(#ifdef VECTOR_SUB_GROUP_BROADCAT
)"
R"(        dequantizeBlockAccum_ns_sgbroadcast_8_lo(totalSum, as_ushort8(regA), regS, regB);
)"
R"(#else
)"
R"(        dequantizeBlockAccum_ns_sgbroadcast_1_lo(totalSum, as_ushort8(regA), regS, regB);
)"
R"(#endif // VECTOR_SUB_GROUP_BROADCAT
)"
R"(    }
)"
R"(
)"
R"(    // reduction in local memory, assumes #wave=4
)"
R"(    __local float2 reduceLM[SIMDGROUP_WIDTH * 3];
)"
R"(    if (groupId == 1) reduceLM[SIMDGROUP_WIDTH * 0 + slid] = totalSum;
)"
R"(    if (groupId == 2) reduceLM[SIMDGROUP_WIDTH * 1 + slid] = totalSum;
)"
R"(    if (groupId == 3) reduceLM[SIMDGROUP_WIDTH * 2 + slid] = totalSum;
)"
R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 0 + slid];
)"
R"(    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 1 + slid];
)"
R"(    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 2 + slid];
)"
R"(
)"
R"(    // 2 outputs per fiber in wave 0
)"
R"(    if (groupId == 0) {
)"
R"(        dst = (global float*)((global char*)dst + offsetd);
)"
R"(        vstore2(totalSum, 0, &(dst[gid * 2]));
)"
R"(    }
)"
R"(
)"
R"(}
)"
