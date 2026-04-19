R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(
)"
R"(kernel void kernel_im2col_f32(
)"
R"(        global float * src1,
)"
R"(        ulong offset1,
)"
R"(        global float * dst,
)"
R"(        ulong offsetd,
)"
R"(        ulong batch_offset,
)"
R"(        ulong delta_offset,
)"
R"(        long IW,
)"
R"(        long IH,
)"
R"(        long IC,
)"
R"(        long OW,
)"
R"(        long OH,
)"
R"(        long KW,
)"
R"(        long KH,
)"
R"(        long pelements,
)"
R"(        long CHW,
)"
R"(        int  s0,
)"
R"(        int  s1,
)"
R"(        int  p0,
)"
R"(        int  p1,
)"
R"(        int  d0,
)"
R"(        int  d1
)"
R"() {
)"
R"(    long i = get_global_id(0);
)"
R"(    if (i >= pelements) {
)"
R"(        return;
)"
R"(    }
)"
R"(
)"
R"(    src1 = (global float*)((global char*)src1 + offset1);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    long  ksize = OW * KH;
)"
R"(    long  kx = i / ksize;
)"
R"(    long  kd = kx * ksize;
)"
R"(    long  ky = (i - kd) / OW;
)"
R"(    long  ix = i % OW;
)"
R"(
)"
R"(    long  oh = get_group_id(1);
)"
R"(    long  batch = get_group_id(2) / IC;
)"
R"(    long  ic = get_group_id(2) % IC;
)"
R"(
)"
R"(    long iiw = ix * s0 + kx * d0 - p0;
)"
R"(    long iih = oh * s1 + ky * d1 - p1;
)"
R"(
)"
R"(    long offset_dst =
)"
R"(        ((batch * OH + oh) * OW + ix) * CHW +
)"
R"(        (ic * (KW * KH) + ky * KW + kx);
)"
R"(
)"
R"(    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
)"
R"(        dst[offset_dst] = 0.0f;
)"
R"(    } else {
)"
R"(        long offset_src = ic * delta_offset + batch * batch_offset;
)"
R"(        dst[offset_dst] = src1[offset_src + iih * IW + iiw];
)"
R"(    }
)"
R"(}
)"
