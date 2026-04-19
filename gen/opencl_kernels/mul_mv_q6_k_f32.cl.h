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
		R"(// block_q6_K
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(// 6-bit quantization
)"
		R"(// weight is represented as x = a * q
)"
		R"(// 16 blocks of 16 elements each
)"
		R"(// Effectively 6.5625 bits per weight
)"
		R"(typedef struct {
)"
		R"(    uint8_t ql[QK_K/2];      // quants, lower 4 bits
)"
		R"(    uint8_t qh[QK_K/4];      // quants, upper 2 bits
)"
		R"(    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
)"
		R"(    half d;             // super-block scale
)"
		R"(} block_q6_K;
)"
		R"(
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(// kernel_mul_mv_q6_K_f32
)"
		R"(//------------------------------------------------------------------------------
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
		R"(#define N_DST 1 // number of rows each SIMD group works on
)"
		R"(#define N_SIMDGROUP 2 // number of SIMD groups in a thread group
)"
		R"(#define N_SIMDWIDTH 16 // SIMD group size
)"
		R"(#elif defined (ADRENO_GPU)
)"
		R"(#define N_DST 1
)"
		R"(#define N_SIMDGROUP 2
)"
		R"(#define N_SIMDWIDTH 64
)"
		R"(#endif
)"
		R"(
)"
		R"(#define BLOCK_STRIDE (N_SIMDWIDTH/16) // number of blocks each subgroup processes
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
		R"(kernel void kernel_mul_mv_q6_K_f32(
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
		R"(    uchar kmask1 = 0x03;
)"
		R"(    uchar kmask2 = 0x0C;
)"
		R"(    uchar kmask3 = 0x30;
)"
		R"(    uchar kmask4 = 0xC0;
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
		R"(
)"
		R"(    int row = N_SIMDGROUP * r0 + get_sub_group_id();
)"
		R"(
)"
		R"(    if (row >= ne01) {
)"
		R"(        return;
)"
		R"(    }
)"
		R"(
)"
		R"(    int i12 = im%ne12;
)"
		R"(    int i13 = im/ne12;
)"
		R"(
)"
		R"(    ulong offset_src0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
)"
		R"(
)"
		R"(    global block_q6_K * x = (global block_q6_K *) src0 + row*nb + offset_src0;
)"
		R"(    global float      * yy = (global float     *) src1 + r1*ne10 + im*ne00*ne1;
)"
		R"(
)"
		R"(    float sumf = 0;
)"
		R"(
)"
		R"(    // For Q6_K quantization, 16 values forms a subblock, 16 subblock forms a
)"
		R"(    // block. Values in a subblock shares a scale that is quantized with 8 bits;
)"
		R"(    // the entire block shares a single floating point scale.
)"
		R"(    // For work distribution, each thread processes a subblock (16 weights), hence
)"
		R"(    // 16 threads process a (super) block -- a subgroup thus handles SIMDWIDTH/16
)"
		R"(    // (super) blocks -- this is the block stride.
)"
		R"(    // The 16 threads that process a (super) block are split into 2 portions, each has
)"
		R"(    // 8 threads; each portion works on 8 subblocks.
)"
		R"(    // For subgroup of 16 threads, the entire subgroup works on a single (super) block
)"
		R"(    // before moving to the next (super) block. Thread0 - thread7 work on the
)"
		R"(    // first 8 subblocks; thread8 - thread15 works on the last 8 subblocks.
)"
		R"(    // Thread0 - thread3 work on subblocks 0, 2, 4, 6; thread4 - thread7 work on
)"
		R"(    // subblocks 1, 3, 5, 7. Each thread does not work on an entire subblock, but
)"
		R"(    // works on a total of 16 weight values.
)"
		R"(    int tid  = get_sub_group_local_id()/BLOCK_STRIDE; // first block_stride groups have tid=0
)"
		R"(    int ix   = get_sub_group_local_id()%BLOCK_STRIDE; // first block is 0..block_stride-1
)"
		R"(    int ip   = tid/8;   // first or second half of (super) block (0 or 1)
)"
		R"(    int il   = tid%8;   // each half has 8 parts, one per scale
)"
		R"(    int n    = 4;       // 4 scales at a time (and 4 sums)
)"
		R"(    int l0   = n*il;    // offset into half-block, 0..28
)"
		R"(    int is   = 8*ip + l0/16; // 0, 1, 8, 9
)"
		R"(
)"
		R"(    int y_offset = 128*ip + l0;
)"
		R"(    int q_offset_l = 64*ip + l0;
)"
		R"(    int q_offset_h = 32*ip + l0;
)"
		R"(
)"
		R"(    for (int i = ix; i < nb; i += BLOCK_STRIDE) {
)"
		R"(
)"
		R"(        global uint8_t * q1 = x[i].ql + q_offset_l;
)"
		R"(        global uint8_t * q2 = q1 + QK_K/8;
)"
		R"(        global uint8_t * qh = x[i].qh + q_offset_h;
)"
		R"(        global int8_t  * sc = x[i].scales + is;
)"
		R"(
)"
		R"(        global float * y = yy + i * QK_K + y_offset;
)"
		R"(
)"
		R"(        float dall = x[i].d;
)"
		R"(
)"
		R"(        float4 sums = {0.f, 0.f, 0.f, 0.f};
)"
		R"(
)"
		R"(        sums.s0 += y[0+ 0] * ((float)((q1[0] & 0xF) | ((qh[0] & kmask1) << 4)) - 32.f);
)"
		R"(        sums.s1 += y[0+32] * ((float)((q2[0] & 0xF) | ((qh[0] & kmask2) << 2)) - 32.f);
)"
		R"(        sums.s2 += y[0+64] * ((float)((q1[0]  >> 4) | ((qh[0] & kmask3) << 0)) - 32.f);
)"
		R"(        sums.s3 += y[0+96] * ((float)((q2[0]  >> 4) | ((qh[0] & kmask4) >> 2)) - 32.f);
)"
		R"(
)"
		R"(        sums.s0 += y[1+ 0] * ((float)((q1[1] & 0xF) | ((qh[1] & kmask1) << 4)) - 32.f);
)"
		R"(        sums.s1 += y[1+32] * ((float)((q2[1] & 0xF) | ((qh[1] & kmask2) << 2)) - 32.f);
)"
		R"(        sums.s2 += y[1+64] * ((float)((q1[1]  >> 4) | ((qh[1] & kmask3) << 0)) - 32.f);
)"
		R"(        sums.s3 += y[1+96] * ((float)((q2[1]  >> 4) | ((qh[1] & kmask4) >> 2)) - 32.f);
)"
		R"(
)"
		R"(        sums.s0 += y[2+ 0] * ((float)((q1[2] & 0xF) | ((qh[2] & kmask1) << 4)) - 32.f);
)"
		R"(        sums.s1 += y[2+32] * ((float)((q2[2] & 0xF) | ((qh[2] & kmask2) << 2)) - 32.f);
)"
		R"(        sums.s2 += y[2+64] * ((float)((q1[2]  >> 4) | ((qh[2] & kmask3) << 0)) - 32.f);
)"
		R"(        sums.s3 += y[2+96] * ((float)((q2[2]  >> 4) | ((qh[2] & kmask4) >> 2)) - 32.f);
)"
		R"(
)"
		R"(        sums.s0 += y[3+ 0] * ((float)((q1[3] & 0xF) | ((qh[3] & kmask1) << 4)) - 32.f);
)"
		R"(        sums.s1 += y[3+32] * ((float)((q2[3] & 0xF) | ((qh[3] & kmask2) << 2)) - 32.f);
)"
		R"(        sums.s2 += y[3+64] * ((float)((q1[3]  >> 4) | ((qh[3] & kmask3) << 0)) - 32.f);
)"
		R"(        sums.s3 += y[3+96] * ((float)((q2[3]  >> 4) | ((qh[3] & kmask4) >> 2)) - 32.f);
)"
		R"(
)"
		R"(        sumf += dall * (sums.s0 * sc[0] + sums.s1 * sc[2] + sums.s2 * sc[4] + sums.s3 * sc[6]);
)"
		R"(    }
)"
		R"(
)"
		R"(    float tot = sub_group_reduce_add(sumf);
)"
		R"(    if (get_sub_group_local_id() == 0) {
)"
		R"(        dst[r1*ne0 + im*ne0*ne1 + row] = tot;
)"
		R"(    }
)"
		R"(}
)"
