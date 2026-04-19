R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// tri
)"
R"(//------------------------------------------------------------------------------
)"
R"(__kernel void kernel_tri_f32(
)"
R"(        global float * src0,
)"
R"(        ulong offset0,
)"
R"(        global float * dst,
)"
R"(        ulong offsetd,
)"
R"(        int n,
)"
R"(        int ne0,
)"
R"(        int ne1,
)"
R"(        int tri_type
)"
R"() {
)"
R"(    src0 = (global float*)((global char*)src0 + offset0);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    int idx = get_global_id(0);
)"
R"(    if (idx >= n) return;
)"
R"(
)"
R"(    int i0 = idx % ne0;
)"
R"(    int i1 = (idx / ne0) % ne1;
)"
R"(
)"
R"(    int keep = 0;
)"
R"(    if (tri_type == 0) keep = (i0 >= i1);
)"
R"(    else if (tri_type == 1) keep = (i0 >  i1);
)"
R"(    else if (tri_type == 2) keep = (i0 <= i1);
)"
R"(    else                    keep = (i0 <  i1);
)"
R"(
)"
R"(    dst[idx] = keep ? src0[idx] : 0.0f;
)"
R"(}
)"
