R"(kernel void kernel_upscale(
)"
R"(    global const void * p_src0,
)"
R"(    ulong off_src0,
)"
R"(    global void * p_dst,
)"
R"(    ulong off_dst,
)"
R"(    ulong nb00,
)"
R"(    ulong nb01,
)"
R"(    ulong nb02,
)"
R"(    ulong nb03,
)"
R"(    int ne10,
)"
R"(    int ne11,
)"
R"(    int ne12,
)"
R"(    int ne13,
)"
R"(    float sf0,
)"
R"(    float sf1,
)"
R"(    float sf2,
)"
R"(    float sf3
)"
R"() {
)"
R"(    global const char * src_base = (global const char *)p_src0 + off_src0;
)"
R"(    global float * dst_base = (global float *)((global char *)p_dst + off_dst);
)"
R"(
)"
R"(    int index = get_global_id(0);
)"
R"(    int dst_total_elements = ne10 * ne11 * ne12 * ne13;
)"
R"(
)"
R"(    if (index >= dst_total_elements) {
)"
R"(        return;
)"
R"(    }
)"
R"(
)"
R"(    int i10 = index % ne10;
)"
R"(    int i11 = (index / ne10) % ne11;
)"
R"(    int i12 = (index / (ne10 * ne11)) % ne12;
)"
R"(    int i13 = index / (ne10 * ne11 * ne12);
)"
R"(
)"
R"(    int i00 = (int)(i10 / sf0);
)"
R"(    int i01 = (int)(i11 / sf1);
)"
R"(    int i02 = (int)(i12 / sf2);
)"
R"(    int i03 = (int)(i13 / sf3);
)"
R"(
)"
R"(    ulong offset_src_element = (ulong)i03 * nb03 + (ulong)i02 * nb02 + (ulong)i01 * nb01 + (ulong)i00 * nb00;
)"
R"(    global const float * src_element_ptr = (global const float *)(src_base + offset_src_element);
)"
R"(
)"
R"(    dst_base[index] = *src_element_ptr;
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_upscale_bilinear(
)"
R"(    global const void * p_src0,
)"
R"(    ulong off_src0,
)"
R"(    global void * p_dst,
)"
R"(    ulong off_dst,
)"
R"(    ulong nb00,
)"
R"(    ulong nb01,
)"
R"(    ulong nb02,
)"
R"(    ulong nb03,
)"
R"(    int ne00_src,
)"
R"(    int ne01_src,
)"
R"(    int ne10_dst,
)"
R"(    int ne11_dst,
)"
R"(    int ne12_dst,
)"
R"(    int ne13_dst,
)"
R"(    float sf0,
)"
R"(    float sf1,
)"
R"(    float sf2,
)"
R"(    float sf3,
)"
R"(    float pixel_offset
)"
R"() {
)"
R"(    global const char * src_base = (global const char *)p_src0 + off_src0;
)"
R"(    global float * dst_base = (global float *)((global char *)p_dst + off_dst);
)"
R"(
)"
R"(    int index = get_global_id(0);
)"
R"(    int dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
)"
R"(
)"
R"(    if (index >= dst_total_elements) {
)"
R"(        return;
)"
R"(    }
)"
R"(
)"
R"(    int i10_dst = index % ne10_dst;
)"
R"(    int i11_dst = (index / ne10_dst) % ne11_dst;
)"
R"(    int i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
)"
R"(    int i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);
)"
R"(
)"
R"(    int i02_src = (int)(i12_dst / sf2);
)"
R"(    int i03_src = (int)(i13_dst / sf3);
)"
R"(
)"
R"(    float y_src_f = ((float)i11_dst + pixel_offset) / sf1 - pixel_offset;
)"
R"(    long y0_src = (long)floor(y_src_f);
)"
R"(    long y1_src = y0_src + 1;
)"
R"(
)"
R"(    y0_src = max(0L, min(y0_src, (long)ne01_src - 1));
)"
R"(    y1_src = max(0L, min(y1_src, (long)ne01_src - 1));
)"
R"(
)"
R"(    float dy = y_src_f - (float)y0_src;
)"
R"(    dy = max(0.0f, min(dy, 1.0f));
)"
R"(
)"
R"(    float x_src_f = ((float)i10_dst + pixel_offset) / sf0 - pixel_offset;
)"
R"(    long x0_src = (long)floor(x_src_f);
)"
R"(    long x1_src = x0_src + 1;
)"
R"(
)"
R"(    x0_src = max(0L, min(x0_src, (long)ne00_src - 1));
)"
R"(    x1_src = max(0L, min(x1_src, (long)ne00_src - 1));
)"
R"(
)"
R"(    float dx = x_src_f - (float)x0_src;
)"
R"(    dx = max(0.0f, min(dx, 1.0f));
)"
R"(
)"
R"(    global const float * p_a = (global const float *)(src_base + (ulong)x0_src * nb00 + (ulong)y0_src * nb01 + (ulong)i02_src * nb02 + (ulong)i03_src * nb03);
)"
R"(    global const float * p_b = (global const float *)(src_base + (ulong)x1_src * nb00 + (ulong)y0_src * nb01 + (ulong)i02_src * nb02 + (ulong)i03_src * nb03);
)"
R"(    global const float * p_c = (global const float *)(src_base + (ulong)x0_src * nb00 + (ulong)y1_src * nb01 + (ulong)i02_src * nb02 + (ulong)i03_src * nb03);
)"
R"(    global const float * p_d = (global const float *)(src_base + (ulong)x1_src * nb00 + (ulong)y1_src * nb01 + (ulong)i02_src * nb02 + (ulong)i03_src * nb03);
)"
R"(
)"
R"(    const float val_a = *p_a;
)"
R"(    const float val_b = *p_b;
)"
R"(    const float val_c = *p_c;
)"
R"(    const float val_d = *p_d;
)"
R"(
)"
R"(    float result = val_a * (1.0f - dx) * (1.0f - dy) +
)"
R"(                   val_b * dx * (1.0f - dy) +
)"
R"(                   val_c * (1.0f - dx) * dy +
)"
R"(                   val_d * dx * dy;
)"
R"(
)"
R"(    dst_base[index] = result;
)"
R"(}
)"
