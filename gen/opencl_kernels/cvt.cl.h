R"(//------------------------------------------------------------------------------
)"
		R"(// This file is contains kernels for data conversion.
)"
		R"(// These kernels are used when loading the model, so its performance is less
)"
		R"(// important.
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
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
		R"(// block_q4_1
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(struct block_q4_1 {
)"
		R"(    half d; // delta
)"
		R"(    half m; // min
)"
		R"(    uchar qs[QK4_1 / 2]; // nibbles / quants
)"
		R"(};
)"
		R"(
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(// block_q6_K
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(struct block_q6_K {
)"
		R"(    uint8_t ql[QK_K/2];      // quants, lower 4 bits
)"
		R"(    uint8_t qh[QK_K/4];      // quants, upper 2 bits
)"
		R"(    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
)"
		R"(    half d;                  // super-block scale
)"
		R"(};
)"
		R"(
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(// kernel_convert_block_q4_0
)"
		R"(// Convert the block_q4_0 format to 2 separate arrays (AOS -> SOA).
)"
		R"(// This kernel does not deshuffle the bits.
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(kernel void kernel_convert_block_q4_0(
)"
		R"(    global struct block_q4_0 * src0,
)"
		R"(    global uchar * dst_q,
)"
		R"(    global half  * dst_d
)"
		R"() {
)"
		R"(    global struct block_q4_0 * b = (global struct block_q4_0 *) src0 + get_global_id(0);
)"
		R"(    global uchar * q = (global uchar *) dst_q + QK4_0/2*get_global_id(0);
)"
		R"(    global half  * d = (global half *) dst_d + get_global_id(0);
)"
		R"(
)"
		R"(    *d = b->d;
)"
		R"(
)"
		R"(    for (int i = 0; i < QK4_0/2; ++i) {
)"
		R"(        q[i] = b->qs[i];
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_restore_block_q4_0(
)"
		R"(    global uchar * src_q,
)"
		R"(    global half  * src_d,
)"
		R"(    global struct block_q4_0 * dst
)"
		R"() {
)"
		R"(    global struct block_q4_0 * b = (global struct block_q4_0 *) dst + get_global_id(0);
)"
		R"(    global uchar * q = (global uchar *) src_q + QK4_0/2*get_global_id(0);
)"
		R"(    global half  * d = (global half *) src_d + get_global_id(0);
)"
		R"(
)"
		R"(    b->d = *d;
)"
		R"(    for (int i = 0; i < QK4_0/2; ++i) {
)"
		R"(        b->qs[i] = q[i];
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(// kernel_convert_block_q4_0_noshuffle
)"
		R"(// Flatten q4_0 weights and unshuffle the bits
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(
)"
		R"(kernel void kernel_convert_block_q4_0_noshuffle(
)"
		R"(    global struct block_q4_0 * src0,
)"
		R"(    global uchar * dst_q,
)"
		R"(    global half  * dst_d
)"
		R"() {
)"
		R"(    global struct block_q4_0 * b = (global struct block_q4_0 *) src0 + get_global_id(0);
)"
		R"(    global uchar * q = (global uchar *) dst_q + QK4_0/2*get_global_id(0);
)"
		R"(    global half  * d = (global half *) dst_d + get_global_id(0);
)"
		R"(
)"
		R"(    *d = b->d;
)"
		R"(    for (int i = 0; i < QK4_0/4; ++i) {
)"
		R"(        uchar x0 = b->qs[2*i + 0];
)"
		R"(        uchar x1 = b->qs[2*i + 1];
)"
		R"(
)"
		R"(        q[i + 0      ] = convert_uchar(x0 & 0x0F) | convert_uchar((x1 & 0x0F) << 4);
)"
		R"(        q[i + QK4_0/4] = convert_uchar((x0 & 0xF0) >> 4) | convert_uchar(x1 & 0xF0);
)"
		R"(
)"
		R"(#ifdef ADRENO_GPU
)"
		R"(        // Workaround for adreno - must have the following printf statement for
)"
		R"(        // the kernel to work properly. Otherwise it produces incorrect result.
)"
		R"(        // convert_uchar above also seems necessary.
)"
		R"(        // Compare against a large number so that it does not print anything.
)"
		R"(        // get_sub_group_local_id() also works.
)"
		R"(        if (get_global_id(0) == 65536*4096) {
)"
		R"(            printf("%04x - %02x\n", *(global ushort*)d, ((x0 & 0xF0) >> 4) | (x1 & 0xF0));
)"
		R"(        }
)"
		R"(#endif
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_restore_block_q4_0_noshuffle(
)"
		R"(    global uchar * src_q,
)"
		R"(    global half  * src_d,
)"
		R"(    global struct block_q4_0 * dst,
)"
		R"(    uchar mask_0F,
)"
		R"(    uchar mask_F0
)"
		R"() {
)"
		R"(    global struct block_q4_0 * b = (global struct block_q4_0 *) dst + get_global_id(0);
)"
		R"(    global uchar * q = (global uchar *) src_q + QK4_0/2*get_global_id(0);
)"
		R"(    global half  * d = (global half *) src_d + get_global_id(0);
)"
		R"(
)"
		R"(    b->d = *d;
)"
		R"(    for (int i = 0; i < QK4_0/4; ++i) {
)"
		R"(        uchar x0 = q[i + 0      ] ;
)"
		R"(        uchar x1 = q[i + QK4_0/4];
)"
		R"(
)"
		R"(        b->qs[2*i + 0] = convert_uchar((x0 & mask_0F) | ((x1 & mask_0F) << 4));
)"
		R"(        b->qs[2*i + 1] = convert_uchar(((x0 & mask_F0) >> 4) | (x1 & mask_F0));
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(// kernel_convert_block_q4_1
)"
		R"(// Convert the block_q4_1 format to 2 separate arrays (AOS -> SOA).
)"
		R"(// This kernel does not deshuffle the bits.
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(kernel void kernel_convert_block_q4_1(
)"
		R"(    global struct block_q4_1 * src0,
)"
		R"(    global uchar * dst_q,
)"
		R"(    global half  * dst_d,
)"
		R"(    global half  * dst_m
)"
		R"() {
)"
		R"(    global struct block_q4_1 * b = (global struct block_q4_1 *) src0 + get_global_id(0);
)"
		R"(    global uchar * q = (global uchar *) dst_q + QK4_1/2*get_global_id(0);
)"
		R"(    global half  * d = (global half *) dst_d + get_global_id(0);
)"
		R"(    global half  * m = (global half *) dst_m + get_global_id(0);
)"
		R"(
)"
		R"(    *d = b->d;
)"
		R"(    *m = b->m;
)"
		R"(
)"
		R"(    for (int i = 0; i < QK4_1/2; ++i) {
)"
		R"(        q[i] = b->qs[i];
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_restore_block_q4_1(
)"
		R"(    global uchar * src_q,
)"
		R"(    global half  * src_d,
)"
		R"(    global half  * src_m,
)"
		R"(    global struct block_q4_1 * dst
)"
		R"() {
)"
		R"(    global struct block_q4_1 * b = (global struct block_q4_1 *) dst + get_global_id(0);
)"
		R"(    global uchar * q = (global uchar *) src_q + QK4_1/2*get_global_id(0);
)"
		R"(    global half  * d = (global half *) src_d + get_global_id(0);
)"
		R"(    global half  * m = (global half *) src_m + get_global_id(0);
)"
		R"(
)"
		R"(    b->d = *d;
)"
		R"(    b->m = *m;
)"
		R"(    for (int i = 0; i < QK4_1/2; ++i) {
)"
		R"(        b->qs[i] = q[i];
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_convert_block_q4_1_noshuffle(
)"
		R"(    global struct block_q4_1 * src0,
)"
		R"(    global uchar * dst_q,
)"
		R"(    global half  * dst_d,
)"
		R"(    global half  * dst_m
)"
		R"() {
)"
		R"(    global struct block_q4_1 * b = (global struct block_q4_1 *) src0 + get_global_id(0);
)"
		R"(    global uchar * q = (global uchar *) dst_q + QK4_1/2*get_global_id(0);
)"
		R"(    global half  * d = (global half *) dst_d + get_global_id(0);
)"
		R"(    global half  * m = (global half *) dst_m + get_global_id(0);
)"
		R"(
)"
		R"(    *d = b->d;
)"
		R"(    *m = b->m;
)"
		R"(    for (int i = 0; i < QK4_1/4; ++i) {
)"
		R"(        uchar x0 = b->qs[2*i + 0];
)"
		R"(        uchar x1 = b->qs[2*i + 1];
)"
		R"(
)"
		R"(        q[i + 0      ] = convert_uchar(x0 & 0x0F) | convert_uchar((x1 & 0x0F) << 4);
)"
		R"(        q[i + QK4_1/4] = convert_uchar((x0 & 0xF0) >> 4) | convert_uchar(x1 & 0xF0);
)"
		R"(
)"
		R"(#ifdef ADRENO_GPU
)"
		R"(        if (get_global_id(0) == 65536*4096) {
)"
		R"(            printf("%04x - %02x\n", *(global ushort*)d, ((x0 & 0xF0) >> 4) | (x1 & 0xF0));
)"
		R"(        }
)"
		R"(#endif
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_restore_block_q4_1_noshuffle(
)"
		R"(    global uchar * src_q,
)"
		R"(    global half  * src_d,
)"
		R"(    global half  * src_m,
)"
		R"(    global struct block_q4_1 * dst,
)"
		R"(    uchar mask_0F,
)"
		R"(    uchar mask_F0
)"
		R"() {
)"
		R"(    global struct block_q4_1 * b = (global struct block_q4_1 *) dst + get_global_id(0);
)"
		R"(    global uchar * q = (global uchar *) src_q + QK4_1/2*get_global_id(0);
)"
		R"(    global half  * d = (global half *) src_d + get_global_id(0);
)"
		R"(    global half  * m = (global half *) src_m + get_global_id(0);
)"
		R"(
)"
		R"(    b->d = *d;
)"
		R"(    b->m = *m;
)"
		R"(    for (int i = 0; i < QK4_1/4; ++i) {
)"
		R"(        uchar x0 = q[i + 0      ] ;
)"
		R"(        uchar x1 = q[i + QK4_1/4];
)"
		R"(
)"
		R"(        b->qs[2*i + 0] = convert_uchar((x0 & mask_0F) | ((x1 & mask_0F) << 4));
)"
		R"(        b->qs[2*i + 1] = convert_uchar(((x0 & mask_F0) >> 4) | (x1 & mask_F0));
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(// block_mxfp4
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(#define QK_MXFP4 32
)"
		R"(struct block_mxfp4 {
)"
		R"(    uchar e; // E8M0
)"
		R"(    uchar qs[QK_MXFP4 / 2];
)"
		R"(};
)"
		R"(
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(// kernel_convert_block_mxfp4
)"
		R"(// Convert the block_mxfp4 format to 2 separate arrays (AOS -> SOA).
)"
		R"(// This kernel does not deshuffle the bits.
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(kernel void kernel_convert_block_mxfp4(
)"
		R"(    global struct block_mxfp4 * src0,
)"
		R"(    global uchar * dst_q,
)"
		R"(    global uchar * dst_e
)"
		R"() {
)"
		R"(    global struct block_mxfp4 * b = (global struct block_mxfp4 *) src0 + get_global_id(0);
)"
		R"(    global uchar * q = (global uchar *) dst_q + QK_MXFP4 / 2 * get_global_id(0);
)"
		R"(    global uchar * e = (global uchar *) dst_e + get_global_id(0);
)"
		R"(
)"
		R"(    *e = b->e;
)"
		R"(
)"
		R"(    for (int i = 0; i < QK_MXFP4 / 2; ++i) {
)"
		R"(        q[i] = b->qs[i];
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_convert_block_mxfp4_trans(
)"
		R"(    global struct block_mxfp4 * src0,
)"
		R"(    __global uint4 * dst_q,
)"
		R"(    __global uchar * dst_e,
)"
		R"(    uint ne00,
)"
		R"(    uint ne01
)"
		R"() {
)"
		R"(    int i00 = get_global_id(1);
)"
		R"(    uint i01 = get_global_id(0);
)"
		R"(    uint i02 = get_global_id(2);
)"
		R"(
)"
		R"(    uint ne00_blk = ne00 / QK_MXFP4;
)"
		R"(    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
)"
		R"(    uint dst_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;
)"
		R"(
)"
		R"(    global struct block_mxfp4 * b = src0 + src_blk_offset;
)"
		R"(
)"
		R"(    dst_q[dst_blk_offset] = ((global uint4 *)(&(b->qs[0])))[0];
)"
		R"(    dst_e[dst_blk_offset] = b->e;
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_restore_block_mxfp4(
)"
		R"(    global uchar * src_q,
)"
		R"(    global half  * src_e,
)"
		R"(    global struct block_mxfp4 * dst
)"
		R"() {
)"
		R"(    global struct block_mxfp4 * b = (global struct block_mxfp4 *) dst + get_global_id(0);
)"
		R"(    global uchar * q = (global uchar *) src_q + QK_MXFP4 / 2 * get_global_id(0);
)"
		R"(    global uchar * e = (global uchar *) src_e + get_global_id(0);
)"
		R"(
)"
		R"(    b->e = *e;
)"
		R"(    for (int i = 0; i < QK_MXFP4 / 2; ++i) {
)"
		R"(        b->qs[i] = q[i];
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_restore_block_mxfp4_trans(
)"
		R"(    __global uint4 * src_q,
)"
		R"(    __global uchar * src_e,
)"
		R"(    global struct block_mxfp4 * dst,
)"
		R"(    uint ne00,
)"
		R"(    uint ne01
)"
		R"() {
)"
		R"(    int i00 = get_global_id(1);
)"
		R"(    uint i01 = get_global_id(0);
)"
		R"(    uint i02 = get_global_id(2);
)"
		R"(
)"
		R"(    uint ne00_blk = ne00 / QK_MXFP4;
)"
		R"(    uint src_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;
)"
		R"(    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
)"
		R"(
)"
		R"(    global struct block_mxfp4 * b = dst + dst_blk_offset;
)"
		R"(
)"
		R"(    ((global uint4 *)(&(b->qs[0])))[0] = src_q[src_blk_offset];
)"
		R"(    b->e = src_e[src_blk_offset];
)"
		R"(}
)"
		R"(
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(// block_q8_0
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(typedef struct {
)"
		R"(    half d;       // delta
)"
		R"(    char qs[QK8_0]; // quants
)"
		R"(} block_q8_0;
)"
		R"(
)"
		R"(kernel void kernel_convert_block_q8_0(
)"
		R"(    global block_q8_0 * src0,
)"
		R"(    global uchar * dst_q,
)"
		R"(    global half  * dst_d
)"
		R"() {
)"
		R"(    global block_q8_0 * b = (global block_q8_0 *) src0 + get_global_id(0);
)"
		R"(    global uchar      * q = (global uchar *) dst_q + QK8_0*get_global_id(0);
)"
		R"(    global half       * d = (global half *) dst_d + get_global_id(0);
)"
		R"(
)"
		R"(    *d = b->d;
)"
		R"(
)"
		R"(    for (int i = 0; i < QK8_0; ++i) {
)"
		R"(        q[i] = b->qs[i];
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_restore_block_q8_0(
)"
		R"(    global uchar * src_q,
)"
		R"(    global half  * src_d,
)"
		R"(    global block_q8_0 * dst
)"
		R"() {
)"
		R"(    global block_q8_0 * b = (global block_q8_0 *) dst + get_global_id(0);
)"
		R"(    global uchar      * q = (global uchar *) src_q + QK8_0*get_global_id(0);
)"
		R"(    global half       * d = (global half *) src_d + get_global_id(0);
)"
		R"(
)"
		R"(    b->d = *d;
)"
		R"(    for (int i = 0; i < QK8_0; ++i) {
)"
		R"(        b->qs[i] = q[i];
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_restore_block_q8_0_trans(
)"
		R"(    global uchar * src_q,
)"
		R"(    global half  * src_d,
)"
		R"(    global block_q8_0 * dst,
)"
		R"(    uint ne00,
)"
		R"(    uint ne01
)"
		R"(){
)"
		R"(    uint num_blk_per_row = ne00 / QK8_0;
)"
		R"(
)"
		R"(    global block_q8_0 * b = (global block_q8_0 *) dst + get_global_id(0) * num_blk_per_row;
)"
		R"(    global uchar      * q = (global uchar *) src_q + get_global_id(0) * 4; // 4 8-bit packed
)"
		R"(    global half       * d = (global half *) src_d + get_global_id(0);
)"
		R"(
)"
		R"(    for (uint blk = 0; blk < num_blk_per_row; blk++) {
)"
		R"(        b->d = *d;
)"
		R"(
)"
		R"(        for (uint i = 0; i < QK8_0; i+=4) {
)"
		R"(            b->qs[i]   = q[0];
)"
		R"(            b->qs[i+1] = q[1];
)"
		R"(            b->qs[i+2] = q[2];
)"
		R"(            b->qs[i+3] = q[3];
)"
		R"(
)"
		R"(            q += 4 * ne01; // M stride
)"
		R"(        }
)"
		R"(
)"
		R"(        d += ne01;
)"
		R"(
)"
		R"(        b++;
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(// kernel_convert_block_q6_K
)"
		R"(// Convert the block_q6_K format to 3 separate arrays (AOS -> SOA).
)"
		R"(// This kernel does not deshuffle the bits.
)"
		R"(// Each thread processes a super block.
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(kernel void kernel_convert_block_q6_K(
)"
		R"(    global struct block_q6_K * src0,
)"
		R"(    global uchar * dst_ql,
)"
		R"(    global uchar * dst_qh,
)"
		R"(    global char  * dst_s,
)"
		R"(    global half  * dst_d
)"
		R"() {
)"
		R"(    global struct block_q6_K * b = (global struct block_q6_K *) src0 + get_global_id(0);
)"
		R"(    global uchar * ql = (global uchar *) dst_ql + QK_K/2*get_global_id(0);
)"
		R"(    global uchar * qh = (global uchar *) dst_qh + QK_K/4*get_global_id(0);
)"
		R"(    global char  * s  = (global char  *) dst_s  + QK_K/16*get_global_id(0);
)"
		R"(    global half  * d  = (global half  *) dst_d  + get_global_id(0);
)"
		R"(
)"
		R"(    *d = b->d;
)"
		R"(
)"
		R"(    for (int i = 0; i < QK_K/2; ++i) {
)"
		R"(        ql[i] = b->ql[i];
)"
		R"(    }
)"
		R"(    for (int i = 0; i < QK_K/4; ++i) {
)"
		R"(        qh[i] = b->qh[i];
)"
		R"(    }
)"
		R"(    for (int i = 0; i < QK_K/16; ++i) {
)"
		R"(        s[i] = b->scales[i];
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(// Restore block_q6_K from flattened arrays.
)"
		R"(// Each thread processes a super block.
)"
		R"(kernel void kernel_restore_block_q6_K(
)"
		R"(    global uchar * dst_ql,
)"
		R"(    global uchar * dst_qh,
)"
		R"(    global char  * dst_s,
)"
		R"(    global half  * dst_d,
)"
		R"(    global struct block_q6_K * dst
)"
		R"() {
)"
		R"(    global struct block_q6_K * b = (global struct block_q6_K *) dst + get_global_id(0);
)"
		R"(    global uchar * ql = (global uchar *) dst_ql + QK_K/2*get_global_id(0);
)"
		R"(    global uchar * qh = (global uchar *) dst_qh + QK_K/4*get_global_id(0);
)"
		R"(    global char  * s  = (global char  *) dst_s  + QK_K/16*get_global_id(0);
)"
		R"(    global half  * d  = (global half  *) dst_d  + get_global_id(0);
)"
		R"(
)"
		R"(    b->d = *d;
)"
		R"(
)"
		R"(    for (int i = 0; i < QK_K/2; ++i) {
)"
		R"(        b->ql[i] = ql[i];
)"
		R"(    }
)"
		R"(    for (int i = 0; i < QK_K/4; ++i) {
)"
		R"(        b->qh[i] = qh[i];
)"
		R"(    }
)"
		R"(    for (int i = 0; i < QK_K/16; ++i) {
)"
		R"(        b->scales[i] = s[i];
)"
		R"(    }
)"
		R"(}
)"
