R"(kernel void kernel_ssm_conv_f32_f32(
)"
R"(    global char * src0,
)"
R"(    ulong         offset0,
)"
R"(    global char * src1,
)"
R"(    ulong         offset1,
)"
R"(    global char * dst,
)"
R"(    ulong         offsetd,
)"
R"(    ulong         nb00,
)"
R"(    ulong         nb01,
)"
R"(    ulong         nb02,
)"
R"(    int           ne10,
)"
R"(    ulong         nb11,
)"
R"(    ulong         nb0,
)"
R"(    ulong         nb1,
)"
R"(    ulong         nb2
)"
R"(){
)"
R"(    src0 = src0 + offset0;
)"
R"(    src1 = src1 + offset1;
)"
R"(    dst  = dst  + offsetd;
)"
R"(
)"
R"(    int ir = get_global_id(0);
)"
R"(    int i2 = get_global_id(1);
)"
R"(    int i3 = get_global_id(2);
)"
R"(
)"
R"(    int nc  = ne10;
)"
R"(
)"
R"(    global float * s = (global float *) (src0 + ir*nb01 + i2*nb00 + i3*nb02);
)"
R"(    global float * c = (global float *) (src1 + ir*nb11);
)"
R"(    global float * d = (global float *) (dst  + ir*nb0  + i2*nb1  + i3*nb2);
)"
R"(
)"
R"(    float sumf = 0.0f;
)"
R"(
)"
R"(    for (int i0 = 0; i0 < nc; ++i0) {
)"
R"(        sumf += s[i0] * c[i0];
)"
R"(    }
)"
R"(
)"
R"(    d[0] = sumf;
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_ssm_conv_f32_f32_4(
)"
R"(    global char * src0,
)"
R"(    ulong         offset0,
)"
R"(    global char * src1,
)"
R"(    ulong         offset1,
)"
R"(    global char * dst,
)"
R"(    ulong         offsetd,
)"
R"(    ulong         nb00,
)"
R"(    ulong         nb01,
)"
R"(    ulong         nb02,
)"
R"(    int           ne10,
)"
R"(    ulong         nb11,
)"
R"(    ulong         nb0,
)"
R"(    ulong         nb1,
)"
R"(    ulong         nb2
)"
R"() {
)"
R"(    src0 = src0 + offset0;
)"
R"(    src1 = src1 + offset1;
)"
R"(    dst  = dst  + offsetd;
)"
R"(
)"
R"(    int ir = get_global_id(0);
)"
R"(    int i2 = get_global_id(1);
)"
R"(    int i3 = get_global_id(2);
)"
R"(
)"
R"(    int nc = ne10;
)"
R"(
)"
R"(    global float4 * s = (global float4 *) (src0 + ir*nb01 + i2*nb00 + i3*nb02);
)"
R"(    global float4 * c = (global float4 *) (src1 + ir*nb11);
)"
R"(    global float  * d = (global float  *) (dst  + ir*nb0  + i2*nb1  + i3*nb2);
)"
R"(
)"
R"(    float sumf = 0.0f;
)"
R"(
)"
R"(    for (int i0 = 0; i0 < nc/4; ++i0) {
)"
R"(        sumf += dot(s[i0], c[i0]);
)"
R"(    }
)"
R"(
)"
R"(    d[0] = sumf;
)"
R"(}
)"
