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
R"(#define QK8_0 32
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
R"(#define NB_Q8_0 8
)"
R"(
)"
R"(#ifdef INTEL_GPU
)"
R"(#define N_R0_Q8_0 4 // number of rows each subgroup works on
)"
R"(#define N_SG_Q8_0 2 // number of subgroups in a work group
)"
R"(#define N_SIMDWIDTH 16 // subgroup size
)"
R"(#elif defined (ADRENO_GPU)
)"
R"(#define N_R0_Q8_0 4
)"
R"(#define N_SG_Q8_0 2
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
R"(kernel void kernel_mul_mv_q8_0_f32_flat(
)"
R"(    global char * src0_q,
)"
R"(    global half * src0_d,
)"
R"(    global char * src1,
)"
R"(    ulong         offset1,
)"
R"(    global char * dst,
)"
R"(    ulong         offsetd,
)"
R"(    int           ne00,
)"
R"(    int           ne01,
)"
R"(    ulong         nb01,
)"
R"(    ulong         nb02,
)"
R"(    ulong         nb03,
)"
R"(    int           ne12,
)"
R"(    ulong         nb11,
)"
R"(    ulong         nb12,
)"
R"(    ulong         nb13,
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
R"(    src1 = (global char*)((global char*)src1 + offset1);
)"
R"(    dst  = (global char*)((global char*)dst  + offsetd);
)"
R"(
)"
R"(    int nb = ne00/QK8_0;
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
R"(    int first_row = (r0*N_SG_Q8_0 + get_sub_group_id()) * N_R0_Q8_0;
)"
R"(
)"
R"(    uint i12 = im%ne12;
)"
R"(    uint i13 = im/ne12;
)"
R"(
)"
R"(    ulong offset_src1 = r1*nb11 + i12*nb12 + i13*nb13;
)"
R"(    global float * y  = (global float *) (src1 + offset_src1);
)"
R"(
)"
R"(    // pointers to src0 rows
)"
R"(    uint offset_src0_base = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
)"
R"(
)"
R"(    global char * ax0, * ax1, * ax2, * ax3;
)"
R"(    global half * ad0, * ad1, * ad2, * ad3;
)"
R"(    uint offset_src0;
)"
R"(
)"
R"(    offset_src0 = offset_src0_base + 0*nb01;
)"
R"(    offset_src0 = offset_src0/34;
)"
R"(    ax0 = (global char *) ((global char *) src0_q + offset_src0*sizeof(char)*QK8_0);
)"
R"(    ad0 = (global half *) ((global char *) src0_d + offset_src0*sizeof(half));
)"
R"(
)"
R"(    offset_src0 = offset_src0_base + 1*nb01;
)"
R"(    offset_src0 = offset_src0/34;
)"
R"(    ax1 = (global char *) ((global char *) src0_q + offset_src0*sizeof(char)*QK8_0);
)"
R"(    ad1 = (global half *) ((global char *) src0_d + offset_src0*sizeof(half));
)"
R"(
)"
R"(    offset_src0 = offset_src0_base + 2*nb01;
)"
R"(    offset_src0 = offset_src0/34;
)"
R"(    ax2 = (global char *) ((global char *) src0_q + offset_src0*sizeof(char)*QK8_0);
)"
R"(    ad2 = (global half *) ((global char *) src0_d + offset_src0*sizeof(half));
)"
R"(
)"
R"(    offset_src0 = offset_src0_base + 3*nb01;
)"
R"(    offset_src0 = offset_src0/34;
)"
R"(    ax3 = (global char *) ((global char *) src0_q + offset_src0*sizeof(char)*QK8_0);
)"
R"(    ad3 = (global half *) ((global char *) src0_d + offset_src0*sizeof(half));
)"
R"(
)"
R"(    const short ix = get_sub_group_local_id()/4;
)"
R"(    const short il = get_sub_group_local_id()%4;
)"
R"(
)"
R"(    global float * yb = y + ix*QK8_0 + il*NB_Q8_0;
)"
R"(
)"
R"(    float8 yl;
)"
R"(    float8 qv;
)"
R"(    float4 sumf = 0.f;
)"
R"(    float  sumq = 0.f;
)"
R"(    global char * qs;
)"
R"(
)"
R"(    // each thread handles NB_Q8_0 quants at a time
)"
R"(    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/4) {
)"
R"(        yl = vload8(0, yb);
)"
R"(
)"
R"(        qs = ax0 + ib*sizeof(char)*QK8_0 + il*NB_Q8_0;
)"
R"(        qv = convert_float8(vload8(0, qs));
)"
R"(        sumq = 0;
)"
R"(        sumq += qv.s0*yl.s0;
)"
R"(        sumq += qv.s1*yl.s1;
)"
R"(        sumq += qv.s2*yl.s2;
)"
R"(        sumq += qv.s3*yl.s3;
)"
R"(        sumq += qv.s4*yl.s4;
)"
R"(        sumq += qv.s5*yl.s5;
)"
R"(        sumq += qv.s6*yl.s6;
)"
R"(        sumq += qv.s7*yl.s7;
)"
R"(        sumf.s0 += sumq*ad0[ib];
)"
R"(
)"
R"(        qs = ax1 + ib*sizeof(char)*QK8_0 + il*NB_Q8_0;
)"
R"(        qv = convert_float8(vload8(0, qs));
)"
R"(        sumq = 0;
)"
R"(        sumq += qv.s0*yl.s0;
)"
R"(        sumq += qv.s1*yl.s1;
)"
R"(        sumq += qv.s2*yl.s2;
)"
R"(        sumq += qv.s3*yl.s3;
)"
R"(        sumq += qv.s4*yl.s4;
)"
R"(        sumq += qv.s5*yl.s5;
)"
R"(        sumq += qv.s6*yl.s6;
)"
R"(        sumq += qv.s7*yl.s7;
)"
R"(        sumf.s1 += sumq*ad1[ib];
)"
R"(
)"
R"(        qs = ax2 + ib*sizeof(char)*QK8_0 + il*NB_Q8_0;
)"
R"(        qv = convert_float8(vload8(0, qs));
)"
R"(        sumq = 0;
)"
R"(        sumq += qv.s0*yl.s0;
)"
R"(        sumq += qv.s1*yl.s1;
)"
R"(        sumq += qv.s2*yl.s2;
)"
R"(        sumq += qv.s3*yl.s3;
)"
R"(        sumq += qv.s4*yl.s4;
)"
R"(        sumq += qv.s5*yl.s5;
)"
R"(        sumq += qv.s6*yl.s6;
)"
R"(        sumq += qv.s7*yl.s7;
)"
R"(        sumf.s2 += sumq*ad2[ib];
)"
R"(
)"
R"(        qs = ax3 + ib*sizeof(char)*QK8_0 + il*NB_Q8_0;
)"
R"(        qv = convert_float8(vload8(0, qs));
)"
R"(        sumq = 0;
)"
R"(        sumq += qv.s0*yl.s0;
)"
R"(        sumq += qv.s1*yl.s1;
)"
R"(        sumq += qv.s2*yl.s2;
)"
R"(        sumq += qv.s3*yl.s3;
)"
R"(        sumq += qv.s4*yl.s4;
)"
R"(        sumq += qv.s5*yl.s5;
)"
R"(        sumq += qv.s6*yl.s6;
)"
R"(        sumq += qv.s7*yl.s7;
)"
R"(        sumf.s3 += sumq*ad3[ib];
)"
R"(
)"
R"(        yb += N_SIMDWIDTH*NB_Q8_0;
)"
R"(    }
)"
R"(
)"
R"(    global float * dst_f32 = (global float *) dst + (ulong)im*ne0*ne1 + (ulong)r1*ne0;
)"
R"(
)"
R"(    float4 tot = (float4)(
)"
R"(        sub_group_reduce_add(sumf.s0),
)"
R"(        sub_group_reduce_add(sumf.s1),
)"
R"(        sub_group_reduce_add(sumf.s2),
)"
R"(        sub_group_reduce_add(sumf.s3)
)"
R"(    );
)"
R"(
)"
R"(    if (get_sub_group_local_id() == 0) {
)"
R"(        if (first_row + 0 < ne01) {
)"
R"(            dst_f32[first_row + 0] = tot.s0;
)"
R"(        }
)"
R"(        if (first_row + 1 < ne01) {
)"
R"(            dst_f32[first_row + 1] = tot.s1;
)"
R"(        }
)"
R"(        if (first_row + 2 < ne01) {
)"
R"(            dst_f32[first_row + 2] = tot.s2;
)"
R"(        }
)"
R"(        if (first_row + 3 < ne01) {
)"
R"(            dst_f32[first_row + 3] = tot.s3;
)"
R"(        }
)"
R"(    }
)"
R"(}
)"
