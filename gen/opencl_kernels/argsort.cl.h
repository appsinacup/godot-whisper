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
R"(#define SWAP(x, y, T) { T tmp = (x); (x) = (y); (y) = tmp; }
)"
R"(
)"
R"(enum ggml_sort_order {
)"
R"(    GGML_SORT_ORDER_ASC,
)"
R"(    GGML_SORT_ORDER_DESC,
)"
R"(};
)"
R"(
)"
R"(kernel void kernel_argsort_f32_i32(
)"
R"(    global float * src0,
)"
R"(    ulong          offset0,
)"
R"(    global int   * dst,
)"
R"(    ulong          offsetd,
)"
R"(    const int      ne00,
)"
R"(    const int      ne00_pad,
)"
R"(    const int      order,
)"
R"(    local int    * dst_row
)"
R"() {
)"
R"(    // bitonic sort
)"
R"(    int col = get_local_id(0);
)"
R"(    int row = get_group_id(1);
)"
R"(
)"
R"(    if (col >= ne00_pad) {
)"
R"(        return;
)"
R"(    }
)"
R"(
)"
R"(    src0 = (global char  *)((global char *)src0 + offset0);
)"
R"(    dst  = (global float *)((global char *)dst  + offsetd);
)"
R"(
)"
R"(    global float * x_row = src0 + row * ne00;
)"
R"(
)"
R"(    // initialize indices
)"
R"(    dst_row[col] = col;
)"
R"(
)"
R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(
)"
R"(    for (int k = 2; k <= ne00_pad; k *= 2) {
)"
R"(        for (int j = k / 2; j > 0; j /= 2) {
)"
R"(            int ixj = col ^ j;
)"
R"(            if (ixj > col) {
)"
R"(                if ((col & k) == 0) {
)"
R"(                    if (dst_row[col] >= ne00 ||
)"
R"(                        (dst_row[ixj] < ne00 && (order == GGML_SORT_ORDER_ASC ?
)"
R"(                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
)"
R"(                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
)"
R"(                    ) {
)"
R"(                        SWAP(dst_row[col], dst_row[ixj], int);
)"
R"(                    }
)"
R"(                } else {
)"
R"(                    if (dst_row[ixj] >= ne00 ||
)"
R"(                        (dst_row[col] < ne00 && (order == GGML_SORT_ORDER_ASC ?
)"
R"(                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
)"
R"(                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
)"
R"(                    ) {
)"
R"(                        SWAP(dst_row[col], dst_row[ixj], int);
)"
R"(                    }
)"
R"(                }
)"
R"(            }
)"
R"(            barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(        }
)"
R"(    }
)"
R"(
)"
R"(    // copy the result to dst without the padding
)"
R"(    if (col < ne00) {
)"
R"(        dst[row * ne00 + col] = dst_row[col];
)"
R"(    }
)"
R"(}
)"
