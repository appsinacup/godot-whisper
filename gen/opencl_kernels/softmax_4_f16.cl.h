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
R"(#ifdef ADRENO_GPU
)"
R"(REQD_SUBGROUP_SIZE_64
)"
R"(#endif
)"
R"(kernel void kernel_soft_max_4_f16(
)"
R"(        global char * src0,
)"
R"(        ulong offset0,
)"
R"(        global char * src1,
)"
R"(        ulong offset1,
)"
R"(        global char * src2,
)"
R"(        ulong offset2,
)"
R"(        global char * dst,
)"
R"(        ulong offsetd,
)"
R"(        int ne00,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        int ne12,
)"
R"(        int ne13,
)"
R"(        ulong nb11,
)"
R"(        ulong nb12,
)"
R"(        ulong nb13,
)"
R"(        ulong nb1,
)"
R"(        ulong nb2,
)"
R"(        ulong nb3,
)"
R"(        float scale,
)"
R"(        float max_bias,
)"
R"(        float m0,
)"
R"(        float m1,
)"
R"(        int n_head_log2
)"
R"() {
)"
R"(    src0 = src0 + offset0;
)"
R"(    src1 = src1 + offset1;
)"
R"(    src2 = src2 + offset2;
)"
R"(    dst  = dst  + offsetd;
)"
R"(
)"
R"(    int i03 = get_group_id(2);
)"
R"(    int i02 = get_group_id(1);
)"
R"(    int i01 = get_group_id(0);
)"
R"(
)"
R"(    int i13 = i03%ne13;
)"
R"(    int i12 = i02%ne12;
)"
R"(    int i11 = i01;
)"
R"(
)"
R"(    global float4 * psrc4 = (global float4 *)(src0 + i01*nb01 + i02*nb02 + i03*nb03);
)"
R"(    global half4  * pmask = src1 != src0 ? (global half4 *)(src1 + i11*nb11 + i12*nb12 + i13*nb13) : 0;
)"
R"(    global float  * psrc2 = src2 != src0 ? (global float *)(src2) : 0;
)"
R"(    global float4 * pdst4 = (global float4 *)(dst  + i01*nb1 + i02*nb2 + i03*nb3);
)"
R"(
)"
R"(    float slope = 1.0f;
)"
R"(
)"
R"(    // ALiBi
)"
R"(    if (max_bias > 0.0f) {
)"
R"(        int h = i02;
)"
R"(
)"
R"(        float base = h < n_head_log2 ? m0 : m1;
)"
R"(        int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;
)"
R"(
)"
R"(        slope = pow(base, exp);
)"
R"(    }
)"
R"(
)"
R"(    // parallel max
)"
R"(    float4 lmax4 = psrc2 ? psrc2[i02] : -INFINITY;
)"
R"(    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
)"
R"(        lmax4 = fmax(lmax4, psrc4[i00]*scale + slope*(pmask ? convert_float4(pmask[i00]) : 0.0f));
)"
R"(    }
)"
R"(    float lmax = fmax(fmax(lmax4.s0, lmax4.s1), fmax(lmax4.s2, lmax4.s3));
)"
R"(
)"
R"(    const float max = sub_group_reduce_max(lmax);
)"
R"(
)"
R"(    // parallel sum
)"
R"(    float4 lsum4 = 0.0f;
)"
R"(    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
)"
R"(        const float4 exp_psrc4 = exp((psrc4[i00]*scale + slope*(pmask ? convert_float4(pmask[i00]) : 0.0f)) - max);
)"
R"(        lsum4 += exp_psrc4;
)"
R"(        pdst4[i00] = exp_psrc4;
)"
R"(    }
)"
R"(    float lsum = lsum4.s0 + lsum4.s1 + lsum4.s2 + lsum4.s3;
)"
R"(
)"
R"(    float sum = sub_group_reduce_add(lsum);
)"
R"(
)"
R"(    if (psrc2) {
)"
R"(        sum += exp(psrc2[i02] - max);
)"
R"(    }
)"
R"(
)"
R"(    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
)"
R"(        pdst4[i00] /= sum;
)"
R"(    }
)"
R"(}
)"
