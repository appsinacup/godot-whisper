R"(kernel void kernel_repeat_f32(
)"
R"(        global const char * src0,
)"
R"(        ulong               offset0,
)"
R"(        global       char * dst,
)"
R"(        ulong               offsetd,
)"
R"(        int     ne00,
)"
R"(        int     ne01,
)"
R"(        int     ne02,
)"
R"(        int     ne03,
)"
R"(        ulong   nb00,
)"
R"(        ulong   nb01,
)"
R"(        ulong   nb02,
)"
R"(        ulong   nb03,
)"
R"(        int     ne0,
)"
R"(        ulong   nb0,
)"
R"(        ulong   nb1,
)"
R"(        ulong   nb2,
)"
R"(        ulong   nb3
)"
R"() {
)"
R"(    src0 = src0 + offset0;
)"
R"(    dst  = dst  + offsetd;
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
R"(    const int i03 = i3%ne03;
)"
R"(    const int i02 = i2%ne02;
)"
R"(    const int i01 = i1%ne01;
)"
R"(
)"
R"(    global const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
)"
R"(    global       char * dst_ptr  = dst  +  i3*nb3  +  i2*nb2  +  i1*nb1;
)"
R"(
)"
R"(    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
)"
R"(        const int i00 = i0%ne00;
)"
R"(        *((global float *)(dst_ptr + i0*nb0)) = *((global float *)(src0_ptr + i00*nb00));
)"
R"(    }
)"
R"(}
)"
