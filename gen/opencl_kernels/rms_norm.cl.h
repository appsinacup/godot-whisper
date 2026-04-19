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
R"(//------------------------------------------------------------------------------
)"
R"(// rms_norm
)"
R"(//------------------------------------------------------------------------------
)"
R"(// This kernel depends on subgroup size.
)"
R"(#ifdef INTEL_GPU
)"
R"(REQD_SUBGROUP_SIZE_32
)"
R"(#elif defined (ADRENO_GPU)
)"
R"(REQD_SUBGROUP_SIZE_64
)"
R"(#endif
)"
R"(kernel void kernel_rms_norm(
)"
R"(        global void * src0,
)"
R"(        ulong offset0,
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
R"(        int ne03,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        float eps,
)"
R"(        local float * sum // Note, the size depends on number of subgroups
)"
R"() {
)"
R"(    src0 = (global void*)((global char*)src0 + offset0);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
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
R"(    global float4 * x = (global float4 *) ((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01);
)"
R"(    global float * x_scalar = (global float *) x;
)"
R"(    float4 sumf = 0;
)"
R"(    float all_sum = 0;
)"
R"(
)"
R"(    // parallel sum
)"
R"(    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
)"
R"(        sumf += x[i00] * x[i00];
)"
R"(    }
)"
R"(    all_sum = sumf.s0 + sumf.s1 + sumf.s2 + sumf.s3;
)"
R"(    all_sum = sub_group_reduce_add(all_sum);
)"
R"(    if (get_sub_group_local_id() == 0) {
)"
R"(        sum[get_sub_group_id()] = all_sum;
)"
R"(    }
)"
R"(
)"
R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(    // broadcast
)"
R"(    for (uint i = get_local_size(0) / get_max_sub_group_size() / 2; i > 0; i /= 2) {
)"
R"(       if (get_local_id(0) < i) {
)"
R"(           sum[get_local_id(0)] += sum[get_local_id(0) + i];
)"
R"(       }
)"
R"(    }
)"
R"(    if (get_local_id(0) == 0) {
)"
R"(        for (int i = 4 * (ne00 / 4); i < ne00; i++) {
)"
R"(            sum[0] += x_scalar[i];
)"
R"(        }
)"
R"(        sum[0] /= ne00;
)"
R"(    }
)"
R"(
)"
R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(
)"
R"(    const float mean  = sum[0];
)"
R"(    const float scale = 1.0f/sqrt(mean + eps);
)"
R"(
)"
R"(    global float4 * y = (global float4 *) (dst + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);
)"
R"(    global float * y_scalar = (global float *) y;
)"
R"(    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
)"
R"(        y[i00] = x[i00] * scale;
)"
R"(    }
)"
R"(    if (get_local_id(0) == 0) {
)"
R"(        for (int i00 = 4 * (ne00 / 4); i00 < ne00; i00++) {
)"
R"(            y_scalar[i00] = x_scalar[i00] * scale;
)"
R"(        }
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// rms_norm_mul
)"
R"(//------------------------------------------------------------------------------
)"
R"(#ifdef INTEL_GPU
)"
R"(REQD_SUBGROUP_SIZE_32
)"
R"(#elif defined (ADRENO_GPU)
)"
R"(REQD_SUBGROUP_SIZE_64
)"
R"(#endif
)"
R"(kernel void kernel_rms_norm_mul(
)"
R"(        global char * src0,
)"
R"(        ulong offset0,
)"
R"(        global char * src1,
)"
R"(        ulong offset1,
)"
R"(        global char * dst,
)"
R"(        ulong offsetd,
)"
R"(        int ne00,
)"
R"(        int ne01,
)"
R"(        int ne02,
)"
R"(        int ne03,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        int ne10,
)"
R"(        int ne11,
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
R"(        float eps,
)"
R"(        local float * sum
)"
R"() {
)"
R"(    src0 = src0 + offset0;
)"
R"(    src1 = src1 + offset1;
)"
R"(    dst  = dst  + offsetd;
)"
R"(
)"
R"(    // The size of sum is sizeof(float)*subgroup_size.
)"
R"(    // Each subgroup writes its partial sum to this array.
)"
R"(    // So the number of subgroups per workgroup for this kernel cannot exceed the subgroup size.
)"
R"(    // This is generally true -
)"
R"(    // for subgroup size 64, workgroup size should be less than 4096 (the max is usually 1024).
)"
R"(    if (get_sub_group_id() == 0) {
)"
R"(        sum[get_sub_group_local_id()] = 0.0f;
)"
R"(    }
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
R"(    global float4 * x = (global float4 *) (src0 + i03*nb03 + i02*nb02 + i01*nb01);
)"
R"(    global float4 * f = (global float4 *) (src1 + (i03%ne13)*nb13 + (i02%ne12)*nb12 + (i01%ne11)*nb11);
)"
R"(
)"
R"(    float sumf = 0;
)"
R"(
)"
R"(    // parallel sum
)"
R"(    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
)"
R"(        sumf += dot(x[i00], x[i00]);
)"
R"(    }
)"
R"(    sumf = sub_group_reduce_add(sumf);
)"
R"(
)"
R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(
)"
R"(    if (get_sub_group_local_id() == 0) {
)"
R"(        sum[get_sub_group_id()] = sumf;
)"
R"(    }
)"
R"(
)"
R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(
)"
R"(    //for (uint i = get_local_size(0) / get_max_sub_group_size() / 2; i > 0; i /= 2) {
)"
R"(    //   if (get_local_id(0) < i) {
)"
R"(    //       sum[get_local_id(0)] += sum[get_local_id(0) + i];
)"
R"(    //   }
)"
R"(    //}
)"
R"(    //if (get_local_id(0) == 0) {
)"
R"(    //    sum[0] /= ne00;
)"
R"(    //}
)"
R"(
)"
R"(    //barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(
)"
R"(    sumf = sum[get_sub_group_local_id()];
)"
R"(    sumf = sub_group_reduce_add(sumf);
)"
R"(
)"
R"(    float mean  = sumf / ne00;
)"
R"(    float scale = 1.0f/sqrt(mean + eps);
)"
R"(
)"
R"(    global float4 * y = (global float4 *) (dst + i03*nb3 + i02*nb2 + i01*nb1);
)"
R"(    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
)"
R"(        y[i00] = (x[i00] * scale) * f[i00%(ne10/4)];
)"
R"(    }
)"
R"(}
)"
