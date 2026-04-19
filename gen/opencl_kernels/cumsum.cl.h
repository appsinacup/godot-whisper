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
		R"(// max workgroup size is usually 1024, this covers various subgroups sizes
)"
		R"(#define MAX_SUBGROUPS 128
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
		R"(kernel void kernel_cumsum_blk(
)"
		R"(        global char * src0,
)"
		R"(        ulong offset0,
)"
		R"(        global char * tmp,
)"
		R"(        global char * dst,
)"
		R"(        ulong offsetd,
)"
		R"(        int   ne00,
)"
		R"(        int   ne01,
)"
		R"(        int   ne02,
)"
		R"(        int   ne03,
)"
		R"(        ulong nb00,
)"
		R"(        ulong nb01,
)"
		R"(        ulong nb02,
)"
		R"(        ulong nb03,
)"
		R"(        uint net0,
)"
		R"(        uint net1,
)"
		R"(        uint net2
)"
		R"() {
)"
		R"(    src0 = src0 + offset0;
)"
		R"(    dst  = dst + offsetd;
)"
		R"(
)"
		R"(    const int i3 = get_group_id(2);
)"
		R"(    const int i2 = get_group_id(1);
)"
		R"(    const int i1 = get_group_id(0);
)"
		R"(
)"
		R"(    const int nth = get_local_size(0);
)"
		R"(    const int tid = get_local_id(0);
)"
		R"(
)"
		R"(    const uint sg_size = get_sub_group_size();
)"
		R"(    const uint sg_id = get_sub_group_id();
)"
		R"(    const uint sg_lid = get_sub_group_local_id();
)"
		R"(
)"
		R"(    const int ib = i1 / ne01;
)"
		R"(    const int i00 = ib * nth;
)"
		R"(    const int i01 = i1 % ne01;
)"
		R"(    const int i02 = i2;
)"
		R"(    const int i03 = i3;
)"
		R"(
)"
		R"(    global const float * src0_row = (global const float *)(src0 + i03*nb03 + i02*nb02 + i01*nb01);
)"
		R"(    global       float * tmp_row  = (global float *)tmp + net0 * i01 + net0 * net1 * i02 + net0 * net1 * net2 * i03;
)"
		R"(    global       float * dst_row  = (global float *)dst + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
)"
		R"(
)"
		R"(    __local float partial[MAX_SUBGROUPS];
)"
		R"(
)"
		R"(    float v = 0.0f;
)"
		R"(    if (i00 + tid < ne00) {
)"
		R"(        v = src0_row[i00 + tid];
)"
		R"(    }
)"
		R"(
)"
		R"(    float s = sub_group_scan_inclusive_add(v);
)"
		R"(    if (sg_lid == sg_size - 1) {
)"
		R"(        partial[sg_id] = s;
)"
		R"(    }
)"
		R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
		R"(
)"
		R"(    // NB: subgroup size should be larger than number of subgroups
)"
		R"(    // assuming max workgroup size of 1024, subgroup size should be >= 32
)"
		R"(    if (sg_id == 0) {
)"
		R"(        float x = 0.0f;
)"
		R"(        if (sg_lid < get_num_sub_groups()) {
)"
		R"(            x = partial[sg_lid];
)"
		R"(        }
)"
		R"(        float ex = sub_group_scan_exclusive_add(x);
)"
		R"(        if (sg_lid < get_num_sub_groups()) {
)"
		R"(            partial[sg_lid] = ex;
)"
		R"(        }
)"
		R"(    }
)"
		R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
		R"(
)"
		R"(    s += partial[sg_id];
)"
		R"(
)"
		R"(    if (i00 + tid < ne00) {
)"
		R"(        dst_row[i00 + tid] = s;
)"
		R"(    }
)"
		R"(    if (ne00 > nth && tid == nth - 1) {
)"
		R"(        tmp_row[ib] = s;
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(kernel void kernel_cumsum_add(
)"
		R"(        global char * tmp,
)"
		R"(        global char * dst,
)"
		R"(        ulong offsetd,
)"
		R"(        int   ne00,
)"
		R"(        int   ne01,
)"
		R"(        int   ne02,
)"
		R"(        int   ne03,
)"
		R"(        uint nbt0,
)"
		R"(        uint nbt1,
)"
		R"(        uint nbt2,
)"
		R"(        uint nbt3
)"
		R"() {
)"
		R"(    dst  = dst + offsetd;
)"
		R"(
)"
		R"(    const int i3 = get_group_id(2);
)"
		R"(    const int i2 = get_group_id(1);
)"
		R"(    const int i1 = get_group_id(0);
)"
		R"(
)"
		R"(    const int nth = get_local_size(0);
)"
		R"(    const int tid = get_local_id(0);
)"
		R"(
)"
		R"(    const int ib = i1 / ne01;
)"
		R"(    if (ib == 0) {
)"
		R"(        return;
)"
		R"(    }
)"
		R"(    const int i00 = ib * nth;
)"
		R"(    const int i01 = i1 % ne01;
)"
		R"(    const int i02 = i2;
)"
		R"(    const int i03 = i3;
)"
		R"(
)"
		R"(    global float * tmp_row  = (global float *)(tmp + nbt1 * i01 + nbt2 * i02 + nbt3 * i03);
)"
		R"(    global float * dst_row  = (global float *)dst + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
)"
		R"(
)"
		R"(    if (i00 + tid < ne00) {
)"
		R"(        dst_row[i00 + tid] += tmp_row[ib - 1];
)"
		R"(    }
)"
		R"(}
)"
