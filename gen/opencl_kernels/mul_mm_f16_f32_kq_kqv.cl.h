R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(#pragma OPENCL EXTENSION cl_khr_subgroups : enable
)"
R"(
)"
R"(#define LM_FIRST_256B   0
)"
R"(#define LM_SECOND_256B  64
)"
R"(#define LM_THIRD_256B   128
)"
R"(#define LM_FOURTH_256B  192
)"
R"(
)"
R"(
)"
R"(inline float16 mm_load_a(
)"
R"(    image1d_buffer_t matrix_A,
)"
R"(    uint subMatrixAStartInElements,
)"
R"(    int nb01,
)"
R"(    int line_stride_matrix_A_in_bytes
)"
R"() {
)"
R"(    __private float8 regA;
)"
R"(    size_t sub_block_id_m = get_local_id(0);
)"
R"(
)"
R"(#ifdef KQV
)"
R"(    uint a_texCoord = subMatrixAStartInElements/2 + (sub_block_id_m * nb01/4);
)"
R"(#else // KQ
)"
R"(    uint a_texCoord = subMatrixAStartInElements/2 + (sub_block_id_m * line_stride_matrix_A_in_bytes/4);
)"
R"(#endif
)"
R"(
)"
R"(    regA.s0123  = read_imagef(matrix_A, a_texCoord/4);
)"
R"(    regA.s4567  = read_imagef(matrix_A, (a_texCoord+4)/4);
)"
R"(
)"
R"(    return convert_float16(as_half16(regA));
)"
R"(}
)"
R"(
)"
R"(inline float4 alu_32(
)"
R"(    float16 regA,
)"
R"(    __local float4* matrix_B_vec
)"
R"() {
)"
R"(
)"
R"(    __private float4 rC = 0;
)"
R"(    int i = get_sub_group_id() * 64;
)"
R"(
)"
R"(    rC += regA.s0  * matrix_B_vec[i];
)"
R"(    rC += regA.s1  * matrix_B_vec[i + 16];
)"
R"(    rC += regA.s4  * matrix_B_vec[i + 1];
)"
R"(    rC += regA.s5  * matrix_B_vec[i + 17];
)"
R"(    rC += regA.s8  * matrix_B_vec[i + 2];
)"
R"(    rC += regA.s9  * matrix_B_vec[i + 18];
)"
R"(    rC += regA.sc  * matrix_B_vec[i + 3];
)"
R"(    rC += regA.sd  * matrix_B_vec[i + 19];
)"
R"(
)"
R"(    i += 32;
)"
R"(
)"
R"(    rC += regA.s2  * matrix_B_vec[i];
)"
R"(     rC += regA.s3  * matrix_B_vec[i + 16];
)"
R"(    rC += regA.s6  * matrix_B_vec[i + 1];
)"
R"(    rC += regA.s7  * matrix_B_vec[i + 17];
)"
R"(    rC += regA.sa  * matrix_B_vec[i + 2];
)"
R"(    rC += regA.sb  * matrix_B_vec[i + 18];
)"
R"(    rC += regA.se  * matrix_B_vec[i + 3];
)"
R"(    rC += regA.sf  * matrix_B_vec[i + 19];
)"
R"(
)"
R"(    return rC;
)"
R"(}
)"
R"(
)"
R"(inline float16 alu_16(
)"
R"(    float16 regA,
)"
R"(    __local float* matrix_B_local
)"
R"() {
)"
R"(    float16 out;
)"
R"(    __local float4* matrix_B_vec = (__local float4*)matrix_B_local;
)"
R"(
)"
R"(    out.s0123 = alu_32(regA, matrix_B_vec);
)"
R"(    out.s4567 = alu_32(regA, matrix_B_vec + 4);
)"
R"(    out.s89ab = alu_32(regA, matrix_B_vec + 8);
)"
R"(    out.scdef = alu_32(regA, matrix_B_vec + 12);
)"
R"(
)"
R"(    return out;
)"
R"(}
)"
R"(
)"
R"(inline void mm_mad(
)"
R"(    __local float* matrix_B_local,
)"
R"(    float16 regA,
)"
R"(    float8 regB,
)"
R"(    uint b_localOffsetInWords,
)"
R"(    float16* regC0_ptr,
)"
R"(    float16* regC1_ptr
)"
R"() {
)"
R"(    int offset = b_localOffsetInWords + get_sub_group_id() * 256;
)"
R"(
)"
R"(    matrix_B_local[offset + LM_FIRST_256B] = regB.s0;
)"
R"(    matrix_B_local[offset + LM_SECOND_256B] = regB.s1;
)"
R"(    matrix_B_local[offset + LM_THIRD_256B] = regB.s2;
)"
R"(    matrix_B_local[offset + LM_FOURTH_256B] = regB.s3;
)"
R"(
)"
R"(    float16 add0 = alu_16(regA, matrix_B_local);
)"
R"(    *regC0_ptr += add0;
)"
R"(
)"
R"(    matrix_B_local[offset + LM_FIRST_256B] = regB.s4;
)"
R"(    matrix_B_local[offset + LM_SECOND_256B] = regB.s5;
)"
R"(    matrix_B_local[offset + LM_THIRD_256B] = regB.s6;
)"
R"(    matrix_B_local[offset + LM_FOURTH_256B] = regB.s7;
)"
R"(
)"
R"(    float16 add1 = alu_16(regA, matrix_B_local);
)"
R"(    *regC1_ptr += add1;
)"
R"(}
)"
R"(
)"
R"(inline void mm_store_c_N(
)"
R"(    __write_only image1d_buffer_t matrix_C,
)"
R"(    float16 regC0,
)"
R"(    float16 regC1,
)"
R"(    uint subMatrixCStartInElements,
)"
R"(    int line_stride_matrix_C_in_bytes,
)"
R"(    int mask
)"
R"() {
)"
R"(    size_t sub_block_id_m = get_local_id(0);
)"
R"(
)"
R"(    uint strideInWords     = line_stride_matrix_C_in_bytes/4;
)"
R"(    uint c_coordInWords_0  = (subMatrixCStartInElements + sub_block_id_m);
)"
R"(
)"
R"(    uint c_coordInWords_1  = c_coordInWords_0 + 1  * strideInWords;
)"
R"(    uint c_coordInWords_2  = c_coordInWords_0 + 2  * strideInWords;
)"
R"(    uint c_coordInWords_3  = c_coordInWords_0 + 3  * strideInWords;
)"
R"(    uint c_coordInWords_4  = c_coordInWords_0 + 4  * strideInWords;
)"
R"(    uint c_coordInWords_5  = c_coordInWords_0 + 5  * strideInWords;
)"
R"(    uint c_coordInWords_6  = c_coordInWords_0 + 6  * strideInWords;
)"
R"(    uint c_coordInWords_7  = c_coordInWords_0 + 7  * strideInWords;
)"
R"(    uint c_coordInWords_8  = c_coordInWords_0 + 8  * strideInWords;
)"
R"(    uint c_coordInWords_9  = c_coordInWords_0 + 9  * strideInWords;
)"
R"(    uint c_coordInWords_10 = c_coordInWords_0 + 10 * strideInWords;
)"
R"(    uint c_coordInWords_11 = c_coordInWords_0 + 11 * strideInWords;
)"
R"(    uint c_coordInWords_12 = c_coordInWords_0 + 12 * strideInWords;
)"
R"(    uint c_coordInWords_13 = c_coordInWords_0 + 13 * strideInWords;
)"
R"(    uint c_coordInWords_14 = c_coordInWords_0 + 14 * strideInWords;
)"
R"(    uint c_coordInWords_15 = c_coordInWords_0 + 15 * strideInWords;
)"
R"(    uint c_coordInWords_16 = c_coordInWords_0 + 16 * strideInWords;
)"
R"(    uint c_coordInWords_17 = c_coordInWords_0 + 17 * strideInWords;
)"
R"(    uint c_coordInWords_18 = c_coordInWords_0 + 18 * strideInWords;
)"
R"(    uint c_coordInWords_19 = c_coordInWords_0 + 19 * strideInWords;
)"
R"(    uint c_coordInWords_20 = c_coordInWords_0 + 20 * strideInWords;
)"
R"(    uint c_coordInWords_21 = c_coordInWords_0 + 21 * strideInWords;
)"
R"(    uint c_coordInWords_22 = c_coordInWords_0 + 22 * strideInWords;
)"
R"(    uint c_coordInWords_23 = c_coordInWords_0 + 23 * strideInWords;
)"
R"(    uint c_coordInWords_24 = c_coordInWords_0 + 24 * strideInWords;
)"
R"(    uint c_coordInWords_25 = c_coordInWords_0 + 25 * strideInWords;
)"
R"(    uint c_coordInWords_26 = c_coordInWords_0 + 26 * strideInWords;
)"
R"(    uint c_coordInWords_27 = c_coordInWords_0 + 27 * strideInWords;
)"
R"(    uint c_coordInWords_28 = c_coordInWords_0 + 28 * strideInWords;
)"
R"(    uint c_coordInWords_29 = c_coordInWords_0 + 29 * strideInWords;
)"
R"(    uint c_coordInWords_30 = c_coordInWords_0 + 30 * strideInWords;
)"
R"(    uint c_coordInWords_31 = c_coordInWords_0 + 31 * strideInWords;
)"
R"(
)"
R"(    if (mask > 0)  { write_imagef(matrix_C, c_coordInWords_0, regC0.s0);  }
)"
R"(    if (mask > 1)  { write_imagef(matrix_C, c_coordInWords_1, regC0.s1);  }
)"
R"(    if (mask > 2)  { write_imagef(matrix_C, c_coordInWords_2, regC0.s2);  }
)"
R"(    if (mask > 3)  { write_imagef(matrix_C, c_coordInWords_3, regC0.s3);  }
)"
R"(    if (mask > 4)  { write_imagef(matrix_C, c_coordInWords_4, regC0.s4);  }
)"
R"(    if (mask > 5)  { write_imagef(matrix_C, c_coordInWords_5, regC0.s5);  }
)"
R"(    if (mask > 6)  { write_imagef(matrix_C, c_coordInWords_6, regC0.s6);  }
)"
R"(    if (mask > 7)  { write_imagef(matrix_C, c_coordInWords_7, regC0.s7);  }
)"
R"(    if (mask > 8)  { write_imagef(matrix_C, c_coordInWords_8, regC0.s8);  }
)"
R"(    if (mask > 9)  { write_imagef(matrix_C, c_coordInWords_9, regC0.s9);  }
)"
R"(    if (mask > 10) { write_imagef(matrix_C, c_coordInWords_10, regC0.sa); }
)"
R"(    if (mask > 11) { write_imagef(matrix_C, c_coordInWords_11, regC0.sb); }
)"
R"(    if (mask > 12) { write_imagef(matrix_C, c_coordInWords_12, regC0.sc); }
)"
R"(    if (mask > 13) { write_imagef(matrix_C, c_coordInWords_13, regC0.sd); }
)"
R"(    if (mask > 14) { write_imagef(matrix_C, c_coordInWords_14, regC0.se); }
)"
R"(    if (mask > 15) { write_imagef(matrix_C, c_coordInWords_15, regC0.sf); }
)"
R"(    if (mask > 16) { write_imagef(matrix_C, c_coordInWords_16, regC1.s0); }
)"
R"(    if (mask > 17) { write_imagef(matrix_C, c_coordInWords_17, regC1.s1); }
)"
R"(    if (mask > 18) { write_imagef(matrix_C, c_coordInWords_18, regC1.s2); }
)"
R"(    if (mask > 19) { write_imagef(matrix_C, c_coordInWords_19, regC1.s3); }
)"
R"(    if (mask > 20) { write_imagef(matrix_C, c_coordInWords_20, regC1.s4); }
)"
R"(    if (mask > 21) { write_imagef(matrix_C, c_coordInWords_21, regC1.s5); }
)"
R"(    if (mask > 22) { write_imagef(matrix_C, c_coordInWords_22, regC1.s6); }
)"
R"(    if (mask > 23) { write_imagef(matrix_C, c_coordInWords_23, regC1.s7); }
)"
R"(    if (mask > 24) { write_imagef(matrix_C, c_coordInWords_24, regC1.s8); }
)"
R"(    if (mask > 25) { write_imagef(matrix_C, c_coordInWords_25, regC1.s9); }
)"
R"(    if (mask > 26) { write_imagef(matrix_C, c_coordInWords_26, regC1.sa); }
)"
R"(    if (mask > 27) { write_imagef(matrix_C, c_coordInWords_27, regC1.sb); }
)"
R"(    if (mask > 28) { write_imagef(matrix_C, c_coordInWords_28, regC1.sc); }
)"
R"(    if (mask > 29) { write_imagef(matrix_C, c_coordInWords_29, regC1.sd); }
)"
R"(    if (mask > 30) { write_imagef(matrix_C, c_coordInWords_30, regC1.se); }
)"
R"(    if (mask > 31) { write_imagef(matrix_C, c_coordInWords_31, regC1.sf); }
)"
R"(}
)"
R"(
)"
R"(#define TILESIZE_K 16
)"
R"(#define TILESIZE_M 64
)"
R"(#define TILESIZE_N 32
)"
R"(#ifdef KQV
)"
R"(__kernel void mul_mm_f16_f32_kqv(
)"
R"(#else
)"
R"(__kernel void mul_mm_f16_f32_kq(
)"
R"(#endif
)"
R"(        __read_only  image1d_buffer_t matrix_A,
)"
R"(        int offset0,
)"
R"(        __global float* matrix_B,
)"
R"(        int offset1,
)"
R"(        __write_only image1d_buffer_t matrix_C,
)"
R"(        int offsetd,
)"
R"(        int M, int K, int N,
)"
R"(        int D_A,
)"
R"(        int D_B,
)"
R"(        int nb01
)"
R"() {
)"
R"(
)"
R"(    uint block_id_m = get_global_id(1);
)"
R"(    uint block_id_n = get_global_id(2) % ((N+TILESIZE_N-1)/TILESIZE_N);
)"
R"(    uint block_id_d = get_global_id(2) / ((N+TILESIZE_N-1)/TILESIZE_N);
)"
R"(
)"
R"(    __private float16  regA;
)"
R"(    __private float8   regB;
)"
R"(    __private float16 regC0;
)"
R"(    __private float16 regC1;
)"
R"(
)"
R"(    const uint col   = block_id_m * TILESIZE_M;
)"
R"(    const uint row   = block_id_n * TILESIZE_N;
)"
R"(    const uint depth_A = block_id_d / (D_B/D_A);
)"
R"(    const uint depth_B = block_id_d;
)"
R"(
)"
R"(#ifdef KQV
)"
R"(    int line_stride_matrix_A_in_bytes = nb01 * M;
)"
R"(    int line_stride_matrix_B_in_bytes = K * N * 4;
)"
R"(#else
)"
R"(    int line_stride_matrix_A_in_bytes = K * D_A * 2;
)"
R"(    int line_stride_matrix_B_in_bytes = K * D_B * 4;
)"
R"(#endif
)"
R"(
)"
R"(    int line_stride_matrix_C_in_bytes = M * 4;
)"
R"(
)"
R"(    const uint strideAinElements = line_stride_matrix_A_in_bytes / 2;
)"
R"(    const uint strideBinElements = line_stride_matrix_B_in_bytes / 4;
)"
R"(
)"
R"(    size_t sub_block_id_m = get_local_id(0);
)"
R"(
)"
R"(    uint b_localOffsetInWords = (sub_block_id_m/16)*16
)"
R"(                           + ((((sub_block_id_m)>>0)&1)<<2)
)"
R"(                           + ((((sub_block_id_m)>>1)&1)<<3)
)"
R"(                           + ((((sub_block_id_m)>>2)&1)<<0)
)"
R"(                           + ((((sub_block_id_m)>>3)&1)<<1);
)"
R"(
)"
R"(    uint2 b_globalOffsetInWords_xy = {((sub_block_id_m%4)*4), (sub_block_id_m>>2)};
)"
R"(    uint b_globalOffsetInWords00, b_globalOffsetInWords16;
)"
R"(#ifdef KQV
)"
R"(    b_globalOffsetInWords00 = b_globalOffsetInWords_xy.x + b_globalOffsetInWords_xy.y*K;
)"
R"(    b_globalOffsetInWords16 = b_globalOffsetInWords00 + (16 * K);
)"
R"(    uint subMatrixAStartInElements = depth_A * strideAinElements + col * nb01 / 2;
)"
R"(    uint subMatrixBStartInElements = depth_B * strideBinElements + row * K;
)"
R"(#else
)"
R"(    b_globalOffsetInWords00 = b_globalOffsetInWords_xy.x + b_globalOffsetInWords_xy.y*line_stride_matrix_B_in_bytes/4;
)"
R"(    b_globalOffsetInWords16 = b_globalOffsetInWords00 + (16 * line_stride_matrix_B_in_bytes/4);
)"
R"(    uint subMatrixAStartInElements = col * strideAinElements + depth_A * K;
)"
R"(    uint subMatrixBStartInElements = row * strideBinElements + depth_B * K;
)"
R"(#endif
)"
R"(
)"
R"(    __local float matrix_B_local[1024];
)"
R"(
)"
R"(    for (uint step=0; step < K; step+=TILESIZE_K) {
)"
R"(        size_t sub_block_id_m = get_local_id(0);
)"
R"(        regA = mm_load_a(matrix_A, subMatrixAStartInElements, nb01, line_stride_matrix_A_in_bytes);
)"
R"(
)"
R"(        uint b_coordInWords00 = subMatrixBStartInElements + b_globalOffsetInWords00;
)"
R"(        uint b_coordInWords16 = subMatrixBStartInElements + b_globalOffsetInWords16;
)"
R"(
)"
R"(        regB.s0123 = vload4(b_coordInWords00/4, matrix_B);
)"
R"(        regB.s4567 = vload4(b_coordInWords16/4, matrix_B);
)"
R"(
)"
R"(        mm_mad(matrix_B_local, regA, regB, b_localOffsetInWords, &regC0, &regC1);
)"
R"(
)"
R"(        subMatrixAStartInElements += TILESIZE_K;
)"
R"(        subMatrixBStartInElements += TILESIZE_K;
)"
R"(    }
)"
R"(
)"
R"(    uint subMatrixCStartInElements = depth_B * N * M + row * M + col;
)"
R"(    mm_store_c_N(matrix_C, regC0, regC1, subMatrixCStartInElements, line_stride_matrix_C_in_bytes, (N-block_id_n*32));
)"
R"(}
)"
R"(
)"
