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
		R"(kernel void kernel_l2_norm_f32(
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
		R"(    global float * x = (global float *) ((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01);
)"
		R"(    global float * y = (global float *) (dst + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);
)"
		R"(
)"
		R"(    float sumf = 0;
)"
		R"(
)"
		R"(    // parallel sum
)"
		R"(    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
)"
		R"(        sumf += x[i00] * x[i00];
)"
		R"(    }
)"
		R"(    sumf = sub_group_reduce_add(sumf);
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
		R"(
)"
		R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
		R"(
)"
		R"(    const float scale = 1.0f/max(sqrt(sum[0]), eps);
)"
		R"(
)"
		R"(    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
)"
		R"(        y[i00] = x[i00] * scale;
)"
		R"(    }
)"
		R"(}
)"
