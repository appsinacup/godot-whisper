R"(kernel void kernel_diag_f32(
)"
R"(    global const char * src0,
)"
R"(    ulong               offset0,
)"
R"(    global       char * dst,
)"
R"(    ulong               offsetd,
)"
R"(    ulong               nb01,
)"
R"(    ulong               nb02,
)"
R"(    ulong               nb03,
)"
R"(    int                 ne0,
)"
R"(    ulong               nb0,
)"
R"(    ulong               nb2,
)"
R"(    ulong               nb3
)"
R"() {
)"
R"(    src0 = src0 + offset0;
)"
R"(    dst  = dst + offsetd;
)"
R"(
)"
R"(    int i3 = get_group_id(2);
)"
R"(    int i2 = get_group_id(1);
)"
R"(    int i1 = get_group_id(0);
)"
R"(
)"
R"(    global const float * src0_ptr = (global const float *)(src0 +           i2*nb02 + i3*nb03);
)"
R"(    global       float * dst_ptr  = (global       float *)(dst  + i1*nb01 + i2*nb2  + i3*nb3);
)"
R"(
)"
R"(    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
)"
R"(        dst_ptr[i0] = i0 == i1 ? src0_ptr[i0] : 0.0f;
)"
R"(    }
)"
R"(}
)"
