R"(kernel void kernel_pad(
)"
		R"(        global void * src0,
)"
		R"(        ulong offset0,
)"
		R"(        global void * dst,
)"
		R"(        ulong offsetd,
)"
		R"(        int ne00, int ne01, int ne02, int ne03,
)"
		R"(        ulong nb00, ulong nb01, ulong nb02, ulong nb03,
)"
		R"(        int ne0, int ne1, int ne2, int ne3,
)"
		R"(        ulong nb0, ulong nb1, ulong nb2, ulong nb3,
)"
		R"(        int lp0, int rp0,
)"
		R"(        int lp1, int rp1,
)"
		R"(        int lp2, int rp2,
)"
		R"(        int lp3, int rp3
)"
		R"() {
)"
		R"(    src0 = (global float*)((global char*)src0 + offset0);
)"
		R"(    dst  = (global float*)((global char*)dst  + offsetd);
)"
		R"(
)"
		R"(    int i0 = get_global_id(0);
)"
		R"(    int i1 = get_group_id(1);
)"
		R"(    int i2 = get_group_id(2) % ne2;
)"
		R"(    int i3 = get_group_id(2) / ne2;
)"
		R"(
)"
		R"(    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
)"
		R"(        return;
)"
		R"(    }
)"
		R"(
)"
		R"(    uint src0_idx = (i3 - lp3)*nb03 + (i2 - lp2)*nb02 + (i1 - lp1)*nb01 + (i0 - lp0)*nb00;
)"
		R"(    uint dst_idx  =         i3*nb3  +         i2*nb2  +         i1*nb1  +         i0*nb0;
)"
		R"(
)"
		R"(    global float * src0_ptr = (global float *)((global char *)src0 + src0_idx);
)"
		R"(    global float * dst_ptr  = (global float *)((global char *)dst  + dst_idx);
)"
		R"(
)"
		R"(    bool in_src_bounds = (i0 >= lp0 && i0 < ne0 - rp0) &&
)"
		R"(                         (i1 >= lp1 && i1 < ne1 - rp1) &&
)"
		R"(                         (i2 >= lp2 && i2 < ne2 - rp2) &&
)"
		R"(                         (i3 >= lp3 && i3 < ne3 - rp3);
)"
		R"(
)"
		R"(    *dst_ptr = in_src_bounds ? *src0_ptr : 0.0f;
)"
		R"(}
)"
