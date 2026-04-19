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
R"(typedef struct {
)"
R"(    uchar e; // E8M0
)"
R"(    uchar qs[QK_MXFP4/2];
)"
R"(} block_mxfp4;
)"
R"(
)"
R"(constant static float kvalues_mxfp4_f[16] = {
)"
R"(    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
)"
R"(};
)"
R"(
)"
R"(static inline float e8m0_to_fp32(uchar x) {
)"
R"(    int bits;
)"
R"(
)"
R"(    if (x == 0) {
)"
R"(        bits = 0x00400000;
)"
R"(    } else {
)"
R"(        bits = (uint) x << 23;
)"
R"(    }
)"
R"(
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
R"(#define N_R0_MXFP4 2
)"
R"(#define N_SG_MXFP4 2
)"
R"(#define N_SIMDWIDTH 64
)"
R"(#endif
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
R"(kernel void kernel_mul_mv_mxfp4_f32(
)"
R"(    global char * src0,
)"
R"(    ulong         offset0,
)"
R"(    global char * src1,
)"
R"(    ulong         offset1,
)"
R"(    global char * dst,
)"
R"(    ulong         offsetd,
)"
R"(    int ne00,
)"
R"(    ulong nb01,
)"
R"(    ulong nb02,
)"
R"(    ulong nb03,
)"
R"(    int ne12,
)"
R"(    ulong nb11,
)"
R"(    ulong nb12,
)"
R"(    ulong nb13,
)"
R"(    int ne0,
)"
R"(    int ne1,
)"
R"(    int r2,
)"
R"(    int r3,
)"
R"(    local  char * shmem
)"
R"() {
)"
R"(    src0 = (global char*)((global char*)src0 + offset0);
)"
R"(    src1 = (global char*)((global char*)src1 + offset1);
)"
R"(    dst  = (global char*)((global char*)dst  + offsetd);
)"
R"(
)"
R"(    local float * shmem_f32 = (local float *) shmem;
)"
R"(    int nb = ne00/QK_MXFP4;
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
R"(    int first_row = (r0 * N_SG_MXFP4 + get_sub_group_id()) * N_R0_MXFP4;
)"
R"(
)"
R"(    uint i12 = im%ne12;
)"
R"(    uint i13 = im/ne12;
)"
R"(
)"
R"(    ulong offset_src0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
)"
R"(    ulong offset_src1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;
)"
R"(
)"
R"(    global block_mxfp4 * x = (global block_mxfp4 *) (src0 + offset_src0);
)"
R"(    global float       * y = (global float       *) (src1 + offset_src1);
)"
R"(
)"
R"(    const short ix = get_sub_group_local_id()/2;  // 0...15
)"
R"(    const short it = get_sub_group_local_id()%2;  // 0 or 1
)"
R"(
)"
R"(    shmem_f32[get_sub_group_local_id()] = kvalues_mxfp4_f[get_sub_group_local_id()%16];
)"
R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(
)"
R"(    float4 yl[4];
)"
R"(    float sumf[N_R0_MXFP4] = {0.f};
)"
R"(
)"
R"(    global float * yb = y + ix * QK_MXFP4 + it * 8;
)"
R"(
)"
R"(    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/2) {
)"
R"(        global float4 * y4 = (global float4 *)yb;
)"
R"(        yl[0] = y4[0];
)"
R"(        yl[1] = y4[4];
)"
R"(        yl[2] = y4[1];
)"
R"(        yl[3] = y4[5];
)"
R"(
)"
R"(        for (short row = 0; row < N_R0_MXFP4; row++) {
)"
R"(            global block_mxfp4 * xb = x + row*nb + ib;
)"
R"(            global uchar       * q2 = (global uchar *)(xb->qs + 8*it);
)"
R"(
)"
R"(            float4 acc1 = yl[0]*(float4)(shmem_f32[q2[0] &  0x0F], shmem_f32[q2[1] &  0x0F], shmem_f32[q2[2] &  0x0F], shmem_f32[q2[3] &  0x0F]);
)"
R"(            float4 acc2 = yl[1]*(float4)(shmem_f32[q2[0] >> 4   ], shmem_f32[q2[1] >> 4   ], shmem_f32[q2[2] >> 4   ], shmem_f32[q2[3] >> 4   ]);
)"
R"(            float4 acc3 = yl[2]*(float4)(shmem_f32[q2[4] &  0x0F], shmem_f32[q2[5] &  0x0F], shmem_f32[q2[6] &  0x0F], shmem_f32[q2[7] &  0x0F]);
)"
R"(            float4 acc4 = yl[3]*(float4)(shmem_f32[q2[4] >> 4   ], shmem_f32[q2[5] >> 4   ], shmem_f32[q2[6] >> 4   ], shmem_f32[q2[7] >> 4   ]);
)"
R"(
)"
R"(            acc1 = (acc1 + acc3) + (acc2 + acc4);
)"
R"(
)"
R"(            sumf[row] += e8m0_to_fp32(xb->e) * ((acc1.s0 + acc1.s1) + (acc1.s2 + acc1.s3));
)"
R"(        }
)"
R"(
)"
R"(        yb += (N_SIMDWIDTH/2) * QK_MXFP4;
)"
R"(    }
)"
R"(
)"
R"(    global float * dst_f32 = (global float *) dst + (ulong)im*ne0*ne1 + (ulong)r1*ne0;
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
