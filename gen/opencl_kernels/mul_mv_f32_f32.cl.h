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
R"(#define N_F32_F32 4
)"
R"(
)"
R"(#ifdef ADRENO_GPU
)"
R"(REQD_SUBGROUP_SIZE_64
)"
R"(#endif
)"
R"(kernel void kernel_mul_mat_f32_f32(
)"
R"(        global char * src0,
)"
R"(        ulong offset0,
)"
R"(        global char * src1,
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
R"(        ulong nb00,
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
R"(        ulong nb10,
)"
R"(        ulong nb11,
)"
R"(        ulong nb12,
)"
R"(        ulong nb13,
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
R"(    src0 = (global char*)((global char*)src0 + offset0);
)"
R"(    src1 = (global char*)((global char*)src1 + offset1);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    int r0 = get_group_id(0);
)"
R"(    int rb = get_group_id(1)*N_F32_F32;
)"
R"(    int im = get_group_id(2);
)"
R"(
)"
R"(    int i12 = im%ne12;
)"
R"(    int i13 = im/ne12;
)"
R"(
)"
R"(    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
)"
R"(
)"
R"(    global float * x = (global float *) (src0 + offset_src0);
)"
R"(
)"
R"(    if (ne00 < 128) {
)"
R"(        for (int row = 0; row < N_F32_F32; ++row) {
)"
R"(            int r1 = rb + row;
)"
R"(            if (r1 >= ne11) {
)"
R"(                break;
)"
R"(            }
)"
R"(
)"
R"(            ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;
)"
R"(
)"
R"(            global float * y = (global float *) (src1 + offset_src1);
)"
R"(
)"
R"(            float sumf = 0;
)"
R"(            for (int i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size()) {
)"
R"(                sumf += (float) x[i] * (float) y[i];
)"
R"(            }
)"
R"(
)"
R"(            float all_sum = sub_group_reduce_add(sumf);
)"
R"(            if (get_sub_group_local_id() == 0) {
)"
R"(                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
)"
R"(            }
)"
R"(        }
)"
R"(    } else {
)"
R"(        global float4 * x4 = (global float4 *)x;
)"
R"(        for (int row = 0; row < N_F32_F32; ++row) {
)"
R"(            int r1 = rb + row;
)"
R"(            if (r1 >= ne11) {
)"
R"(                break;
)"
R"(            }
)"
R"(
)"
R"(            ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;
)"
R"(
)"
R"(            global float  * y  = (global float  *) (src1 + offset_src1);
)"
R"(            global float4 * y4 = (global float4 *) y;
)"
R"(
)"
R"(            float sumf = 0;
)"
R"(            for (int i = get_sub_group_local_id(); i < ne00/4; i += get_max_sub_group_size()) {
)"
R"(                sumf += (float) x4[i].s0 * y4[i].s0;
)"
R"(                sumf += (float) x4[i].s1 * y4[i].s1;
)"
R"(                sumf += (float) x4[i].s2 * y4[i].s2;
)"
R"(                sumf += (float) x4[i].s3 * y4[i].s3;
)"
R"(            }
)"
R"(
)"
R"(            float all_sum = sub_group_reduce_add(sumf);
)"
R"(            if (get_sub_group_local_id() == 0) {
)"
R"(                for (int i = 4*(ne00/4); i < ne00; ++i) {
)"
R"(                    all_sum += (float) x[i] * y[i];
)"
R"(                }
)"
R"(                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
)"
R"(            }
)"
R"(        }
)"
R"(    }
)"
R"(}
)"
