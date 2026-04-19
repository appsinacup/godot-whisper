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
R"(//------------------------------------------------------------------------------
)"
R"(// norm
)"
R"(//------------------------------------------------------------------------------
)"
R"(kernel void kernel_norm(
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
R"(        local float * sum
)"
R"() {
)"
R"(    src0 = (global void*)((global char*)src0 + offset0);
)"
R"(    dst = (global void*)((global char*)dst + offsetd);
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
R"(    global float * x = (global float *) ((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01);
)"
R"(
)"
R"(    // MEAN
)"
R"(    // parallel sum
)"
R"(    sum[get_local_id(0)] = 0.0f;
)"
R"(    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
)"
R"(        sum[get_local_id(0)] += x[i00];
)"
R"(    }
)"
R"(    // reduce
)"
R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(    for (uint i = get_local_size(0)/2; i > 0; i /= 2) {
)"
R"(        if (get_local_id(0) < i) {
)"
R"(            sum[get_local_id(0)] += sum[get_local_id(0) + i];
)"
R"(        }
)"
R"(        barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(    }
)"
R"(    float mean  = sum[0] / ne00;
)"
R"(
)"
R"(    // recenter and VARIANCE
)"
R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(    global float * y = dst + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
)"
R"(    sum[get_local_id(0)] = 0.0f;
)"
R"(    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
)"
R"(        y[i00] = x[i00] - mean;
)"
R"(        sum[get_local_id(0)] += y[i00] * y[i00];
)"
R"(    }
)"
R"(
)"
R"(    // reduce
)"
R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(    for (uint i = get_local_size(0)/2; i > 0; i /= 2) {
)"
R"(        if (get_local_id(0) < i) {
)"
R"(            sum[get_local_id(0)] += sum[get_local_id(0) + i];
)"
R"(        }
)"
R"(        barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(    }
)"
R"(    float variance = sum[0] / ne00;
)"
R"(
)"
R"(    float scale = 1.0f/sqrt(variance + eps);
)"
R"(    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
)"
R"(        y[i00] = y[i00] * scale;
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// norm_mul_add
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
R"(kernel void kernel_norm_mul_add(
)"
R"(        global char * src0_ptr, ulong src0_offset,
)"
R"(        global char * src1_ptr, ulong src1_offset,
)"
R"(        global char * src2_ptr, ulong src2_offset,
)"
R"(        global char * dst_ptr,  ulong dst_offset,
)"
R"(        int ne00, int ne01, int ne02, int ne03,
)"
R"(        ulong nb01, ulong nb02, ulong nb03,
)"
R"(        int ne10, int ne11, int ne12, int ne13,
)"
R"(        ulong nb11, ulong nb12, ulong nb13,
)"
R"(        int ne20, int ne21, int ne22, int ne23,
)"
R"(        ulong nb21, ulong nb22, ulong nb23,
)"
R"(        ulong nbd1, ulong nbd2, ulong nbd3,
)"
R"(        float eps,
)"
R"(        local float2 * sums
)"
R"() {
)"
R"(    const int i03 = get_group_id(2);
)"
R"(    const int i02 = get_group_id(1);
)"
R"(    const int i01 = get_group_id(0);
)"
R"(
)"
R"(    global float4 * x = (global float4 *)(src0_ptr + src0_offset + i01*nb01 + i02*nb02 + i03*nb03);
)"
R"(    global float4 * w = (global float4 *)(src1_ptr + src1_offset + (i01%ne11)*nb11 + (i02%ne12)*nb12 + (i03%ne13)*nb13);
)"
R"(    global float4 * b = (global float4 *)(src2_ptr + src2_offset + (i01%ne21)*nb21 + (i02%ne22)*nb22 + (i03%ne23)*nb23);
)"
R"(    global float4 * y = (global float4 *)(dst_ptr  + dst_offset  + i01*nbd1 + i02*nbd2 + i03*nbd3);
)"
R"(
)"
R"(    float p_sum = 0.0f;
)"
R"(    float p_sum_sq = 0.0f;
)"
R"(
)"
R"(    const int n_chunks = ne00 / 4;
)"
R"(    for (int i00 = get_local_id(0); i00 < n_chunks; i00 += get_local_size(0)) {
)"
R"(        float4 val = x[i00];
)"
R"(        p_sum += val.x + val.y + val.z + val.w;
)"
R"(        p_sum_sq += dot(val, val);
)"
R"(    }
)"
R"(
)"
R"(    p_sum = sub_group_reduce_add(p_sum);
)"
R"(    p_sum_sq = sub_group_reduce_add(p_sum_sq);
)"
R"(
)"
R"(    if (get_sub_group_local_id() == 0) {
)"
R"(        sums[get_sub_group_id()] = (float2)(p_sum, p_sum_sq);
)"
R"(    }
)"
R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(
)"
R"(    if (get_local_id(0) == 0) {
)"
R"(        float sum = 0.0f;
)"
R"(        float sum_sq = 0.0f;
)"
R"(        for (uint i = 0; i < get_num_sub_groups(); ++i) {
)"
R"(            float2 s = sums[i];
)"
R"(            sum += s.x;
)"
R"(            sum_sq += s.y;
)"
R"(        }
)"
R"(
)"
R"(        const float inv_ne00 = 1.0f / (float)ne00;
)"
R"(        const float mean = sum * inv_ne00;
)"
R"(        const float variance = mad(-mean, mean, sum_sq * inv_ne00);
)"
R"(
)"
R"(        sums[0] = (float2)(mean, rsqrt(variance + eps));
)"
R"(    }
)"
R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(
)"
R"(    const float2 mean_scale = sums[0];
)"
R"(    const float mean = mean_scale.x;
)"
R"(    const float scale = mean_scale.y;
)"
R"(    const float neg_mean_scale = -mean * scale;
)"
R"(
)"
R"(    for (int i00 = get_local_id(0); i00 < n_chunks; i00 += get_local_size(0)) {
)"
R"(        const int w_idx = ne10 > 1 ? i00 : 0;
)"
R"(        const int b_idx = ne20 > 1 ? i00 : 0;
)"
R"(        const float4 norm_x = mad(x[i00], (float4)scale, (float4)neg_mean_scale);
)"
R"(        y[i00] = mad(norm_x, w[w_idx], b[b_idx]);
)"
R"(    }
)"
R"(}
)"
