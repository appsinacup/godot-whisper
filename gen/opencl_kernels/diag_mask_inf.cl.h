R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// diag_mask_inf kernels
)"
R"(//------------------------------------------------------------------------------
)"
R"(kernel void kernel_diag_mask_inf(
)"
R"(        global float * src0,
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
R"(        int n_past
)"
R"() {
)"
R"(    src0 = (global float*)((global char*)src0 + offset0);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    int i02 = get_global_id(2);
)"
R"(    int i01 = get_global_id(1);
)"
R"(    int i00 = get_global_id(0);
)"
R"(
)"
R"(    if (i00 > n_past + i01) {
)"
R"(        dst[i02*ne01*ne00 + i01*ne00 + i00] = -INFINITY;
)"
R"(    } else {
)"
R"(        dst[i02*ne01*ne00 + i01*ne00 + i00] = src0[i02*ne01*ne00 + i01*ne00 + i00];
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_diag_mask_inf_8(
)"
R"(        global float4 * src0,
)"
R"(        ulong offset0,
)"
R"(        global float4 * dst,
)"
R"(        ulong offsetd,
)"
R"(        int ne00,
)"
R"(        int ne01,
)"
R"(        int n_past
)"
R"() {
)"
R"(    src0 = (global float4*)((global char*)src0 + offset0);
)"
R"(    dst = (global float4*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    int i = 2*get_global_id(0);
)"
R"(
)"
R"(    dst[i+0] = src0[i+0];
)"
R"(    dst[i+1] = src0[i+1];
)"
R"(    int i4 = 4*i;
)"
R"(    int i02 = i4/(ne00*ne01); i4 -= i02*ne00*ne01;
)"
R"(    int i01 = i4/(ne00);      i4 -= i01*ne00;
)"
R"(    int i00 = i4;
)"
R"(    for (int k = 3; k >= 0; --k) {
)"
R"(        if (i00 + 4 + k <= n_past + i01) {
)"
R"(            break;
)"
R"(        }
)"
R"(        (&dst[i+1])[k] = -INFINITY;
)"
R"(        if (i00 + k > n_past + i01) {
)"
R"(            (&dst[i])[k] = -INFINITY;
)"
R"(        }
)"
R"(    }
)"
R"(}
)"
