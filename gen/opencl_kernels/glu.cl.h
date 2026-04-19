R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(
)"
R"(#define GELU_COEF_A     0.044715f
)"
R"(#define GELU_QUICK_COEF -1.702f
)"
R"(#define SQRT_2_OVER_PI  0.79788456080286535587989211986876f
)"
R"(#define SQRT_2_INV      0.70710678118654752440084436210484f
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// geglu
)"
R"(//------------------------------------------------------------------------------
)"
R"(kernel void kernel_geglu(
)"
R"(    global char * src0,
)"
R"(    ulong  offset0,
)"
R"(    global char * src1,
)"
R"(    ulong  offset1,
)"
R"(    global char * dst,
)"
R"(    ulong  offsetd,
)"
R"(    ulong nb01,
)"
R"(    ulong nb11,
)"
R"(    int ne0,
)"
R"(    ulong nb1,
)"
R"(    int ne00_off,
)"
R"(    int ne10_off
)"
R"() {
)"
R"(    src0 = (global char*)((global char*)src0 + offset0);
)"
R"(    src1 = (global char*)((global char*)src1 + offset1);
)"
R"(    dst  = (global char*)((global char*)dst  + offsetd);
)"
R"(
)"
R"(    global float * src0_row = (global float *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
)"
R"(    global float * src1_row = (global float *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
)"
R"(    global float * dst_row  = (global float *) ((global char *) dst  + get_group_id(0)*nb1);
)"
R"(
)"
R"(    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
)"
R"(        const float x0 = src0_row[i0];
)"
R"(        const float x1 = src1_row[i0];
)"
R"(
)"
R"(        const float gelu = 0.5f*x0*(1.0f + tanh(SQRT_2_OVER_PI*x0*(1.0f + GELU_COEF_A*x0*x0)));
)"
R"(
)"
R"(        dst_row[i0] = gelu*x1;
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_geglu_f16(
)"
R"(    global char * src0,
)"
R"(    ulong  offset0,
)"
R"(    global char * src1,
)"
R"(    ulong  offset1,
)"
R"(    global char * dst,
)"
R"(    ulong  offsetd,
)"
R"(    ulong nb01,
)"
R"(    ulong nb11,
)"
R"(    int ne0,
)"
R"(    ulong nb1,
)"
R"(    int ne00_off,
)"
R"(    int ne10_off
)"
R"() {
)"
R"(    src0 = (global char*)((global char*)src0 + offset0);
)"
R"(    src1 = (global char*)((global char*)src1 + offset1);
)"
R"(    dst  = (global char*)((global char*)dst  + offsetd);
)"
R"(
)"
R"(    global half * src0_row = (global half *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
)"
R"(    global half * src1_row = (global half *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
)"
R"(    global half * dst_row  = (global half *) ((global char *) dst  + get_group_id(0)*nb1);
)"
R"(
)"
R"(    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
)"
R"(        const half x0 = src0_row[i0];
)"
R"(        const half x1 = src1_row[i0];
)"
R"(
)"
R"(        const half gelu = 0.5f*x0*(1.0f + tanh(SQRT_2_OVER_PI*x0*(1.0f + GELU_COEF_A*x0*x0)));
)"
R"(
)"
R"(        dst_row[i0] = gelu*x1;
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// reglu
)"
R"(//------------------------------------------------------------------------------
)"
R"(kernel void kernel_reglu(
)"
R"(    global char * src0,
)"
R"(    ulong  offset0,
)"
R"(    global char * src1,
)"
R"(    ulong  offset1,
)"
R"(    global char * dst,
)"
R"(    ulong  offsetd,
)"
R"(    ulong nb01,
)"
R"(    ulong nb11,
)"
R"(    int ne0,
)"
R"(    ulong nb1,
)"
R"(    int ne00_off,
)"
R"(    int ne10_off
)"
R"() {
)"
R"(    src0 = (global char*)((global char*)src0 + offset0);
)"
R"(    src1 = (global char*)((global char*)src1 + offset1);
)"
R"(    dst  = (global char*)((global char*)dst  + offsetd);
)"
R"(
)"
R"(    global float * src0_row = (global float *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
)"
R"(    global float * src1_row = (global float *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
)"
R"(    global float * dst_row  = (global float *) ((global char *) dst  + get_group_id(0)*nb1);
)"
R"(
)"
R"(    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
)"
R"(        const float x0 = src0_row[i0];
)"
R"(        const float x1 = src1_row[i0];
)"
R"(
)"
R"(        dst_row[i0] = x0*x1*(x0 > 0.0f);
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_reglu_f16(
)"
R"(    global char * src0,
)"
R"(    ulong  offset0,
)"
R"(    global char * src1,
)"
R"(    ulong  offset1,
)"
R"(    global char * dst,
)"
R"(    ulong  offsetd,
)"
R"(    ulong nb01,
)"
R"(    ulong nb11,
)"
R"(    int ne0,
)"
R"(    ulong nb1,
)"
R"(    int ne00_off,
)"
R"(    int ne10_off
)"
R"() {
)"
R"(    src0 = (global char*)((global char*)src0 + offset0);
)"
R"(    src1 = (global char*)((global char*)src1 + offset1);
)"
R"(    dst  = (global char*)((global char*)dst  + offsetd);
)"
R"(
)"
R"(    global half * src0_row = (global half *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
)"
R"(    global half * src1_row = (global half *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
)"
R"(    global half * dst_row  = (global half *) ((global char *) dst  + get_group_id(0)*nb1);
)"
R"(
)"
R"(    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
)"
R"(        const half x0 = src0_row[i0];
)"
R"(        const half x1 = src1_row[i0];
)"
R"(
)"
R"(        dst_row[i0] = x0*x1*(x0 > 0.0f);
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// swiglu
)"
R"(//------------------------------------------------------------------------------
)"
R"(kernel void kernel_swiglu(
)"
R"(    global char * src0,
)"
R"(    ulong  offset0,
)"
R"(    global char * src1,
)"
R"(    ulong  offset1,
)"
R"(    global char * dst,
)"
R"(    ulong  offsetd,
)"
R"(    ulong nb01,
)"
R"(    ulong nb11,
)"
R"(    int ne0,
)"
R"(    ulong nb1,
)"
R"(    int ne00_off,
)"
R"(    int ne10_off
)"
R"() {
)"
R"(    src0 = (global char*)((global char*)src0 + offset0);
)"
R"(    src1 = (global char*)((global char*)src1 + offset1);
)"
R"(    dst  = (global char*)((global char*)dst  + offsetd);
)"
R"(
)"
R"(    global float * src0_row = (global float *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
)"
R"(    global float * src1_row = (global float *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
)"
R"(    global float * dst_row  = (global float *) ((global char *) dst  + get_group_id(0)*nb1);
)"
R"(
)"
R"(    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
)"
R"(        const float x0 = src0_row[i0];
)"
R"(        const float x1 = src1_row[i0];
)"
R"(
)"
R"(        const float silu = x0 / (1.0f + exp(-x0));
)"
R"(
)"
R"(        dst_row[i0] = silu*x1;
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_swiglu_f16(
)"
R"(    global char * src0,
)"
R"(    ulong  offset0,
)"
R"(    global char * src1,
)"
R"(    ulong  offset1,
)"
R"(    global char * dst,
)"
R"(    ulong  offsetd,
)"
R"(    ulong nb01,
)"
R"(    ulong nb11,
)"
R"(    int ne0,
)"
R"(    ulong nb1,
)"
R"(    int ne00_off,
)"
R"(    int ne10_off
)"
R"() {
)"
R"(    src0 = (global char*)((global char*)src0 + offset0);
)"
R"(    src1 = (global char*)((global char*)src1 + offset1);
)"
R"(    dst  = (global char*)((global char*)dst  + offsetd);
)"
R"(
)"
R"(    global half * src0_row = (global half *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
)"
R"(    global half * src1_row = (global half *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
)"
R"(    global half * dst_row  = (global half *) ((global char *) dst  + get_group_id(0)*nb1);
)"
R"(
)"
R"(    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
)"
R"(        const half x0 = src0_row[i0];
)"
R"(        const half x1 = src1_row[i0];
)"
R"(
)"
R"(        const half silu = x0 / (1.0f + exp(-x0));
)"
R"(
)"
R"(        dst_row[i0] = silu*x1;
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// swiglu_oai
)"
R"(//------------------------------------------------------------------------------
)"
R"(kernel void kernel_swiglu_oai(
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
R"(    ulong         nb01,
)"
R"(    ulong         nb11,
)"
R"(    int           ne0,
)"
R"(    ulong         nb1,
)"
R"(    int           ne00_off,
)"
R"(    int           ne10_off,
)"
R"(    float         limit,
)"
R"(    float         alpha
)"
R"() {
)"
R"(    src0 = (global char*)((global char*)src0 + offset0);
)"
R"(    src1 = (global char*)((global char*)src1 + offset1);
)"
R"(    dst  = (global char*)((global char*)dst  + offsetd);
)"
R"(
)"
R"(    global float * src0_row = (global float *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
)"
R"(    global float * src1_row = (global float *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
)"
R"(    global float * dst_row  = (global float *) ((global char *) dst  + get_group_id(0)*nb1);
)"
R"(
)"
R"(    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
)"
R"(        float x0 = src0_row[i0];
)"
R"(        float x1 = src1_row[i0];
)"
R"(
)"
R"(        x0 = min(x0, limit);
)"
R"(        x1 = max(min(x1, limit), -limit);
)"
R"(
)"
R"(        float out_glu = x0 / (1.0f + exp(-x0 * alpha));
)"
R"(        out_glu = out_glu * (1.0f + x1);
)"
R"(
)"
R"(        dst_row[i0] = out_glu;
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// geglu_erf
)"
R"(//------------------------------------------------------------------------------
)"
R"(kernel void kernel_geglu_erf(
)"
R"(    global char * src0,
)"
R"(    ulong  offset0,
)"
R"(    global char * src1,
)"
R"(    ulong  offset1,
)"
R"(    global char * dst,
)"
R"(    ulong  offsetd,
)"
R"(    ulong nb01,
)"
R"(    ulong nb11,
)"
R"(    int ne0,
)"
R"(    ulong nb1,
)"
R"(    int ne00_off,
)"
R"(    int ne10_off
)"
R"() {
)"
R"(    src0 = (global char*)((global char*)src0 + offset0);
)"
R"(    src1 = (global char*)((global char*)src1 + offset1);
)"
R"(    dst  = (global char*)((global char*)dst  + offsetd);
)"
R"(
)"
R"(    global float * src0_row = (global float *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
)"
R"(    global float * src1_row = (global float *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
)"
R"(    global float * dst_row  = (global float *) ((global char *) dst  + get_group_id(0)*nb1);
)"
R"(
)"
R"(    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
)"
R"(        const float x0 = src0_row[i0];
)"
R"(        const float x1 = src1_row[i0];
)"
R"(
)"
R"(        const float gelu_erf = 0.5f*x0*(1.0f + erf(x0*SQRT_2_INV));
)"
R"(
)"
R"(        dst_row[i0] = gelu_erf*x1;
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_geglu_erf_f16(
)"
R"(    global char * src0,
)"
R"(    ulong  offset0,
)"
R"(    global char * src1,
)"
R"(    ulong  offset1,
)"
R"(    global char * dst,
)"
R"(    ulong  offsetd,
)"
R"(    ulong nb01,
)"
R"(    ulong nb11,
)"
R"(    int ne0,
)"
R"(    ulong nb1,
)"
R"(    int ne00_off,
)"
R"(    int ne10_off
)"
R"() {
)"
R"(    src0 = (global char*)((global char*)src0 + offset0);
)"
R"(    src1 = (global char*)((global char*)src1 + offset1);
)"
R"(    dst  = (global char*)((global char*)dst  + offsetd);
)"
R"(
)"
R"(    global half * src0_row = (global half *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
)"
R"(    global half * src1_row = (global half *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
)"
R"(    global half * dst_row  = (global half *) ((global char *) dst  + get_group_id(0)*nb1);
)"
R"(
)"
R"(    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
)"
R"(        const half x0 = src0_row[i0];
)"
R"(        const half x1 = src1_row[i0];
)"
R"(
)"
R"(        const half gelu_erf = 0.5f*x0*(1.0f + erf(x0*SQRT_2_INV));
)"
R"(
)"
R"(        dst_row[i0] = gelu_erf*x1;
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// geglu_quick
)"
R"(//------------------------------------------------------------------------------
)"
R"(kernel void kernel_geglu_quick(
)"
R"(    global char * src0,
)"
R"(    ulong  offset0,
)"
R"(    global char * src1,
)"
R"(    ulong  offset1,
)"
R"(    global char * dst,
)"
R"(    ulong  offsetd,
)"
R"(    ulong nb01,
)"
R"(    ulong nb11,
)"
R"(    int ne0,
)"
R"(    ulong nb1,
)"
R"(    int ne00_off,
)"
R"(    int ne10_off
)"
R"() {
)"
R"(    src0 = (global char*)((global char*)src0 + offset0);
)"
R"(    src1 = (global char*)((global char*)src1 + offset1);
)"
R"(    dst  = (global char*)((global char*)dst  + offsetd);
)"
R"(
)"
R"(    global float * src0_row = (global float *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
)"
R"(    global float * src1_row = (global float *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
)"
R"(    global float * dst_row  = (global float *) ((global char *) dst  + get_group_id(0)*nb1);
)"
R"(
)"
R"(    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
)"
R"(        const float x0 = src0_row[i0];
)"
R"(        const float x1 = src1_row[i0];
)"
R"(
)"
R"(        const float gelu_quick = x0*(1.0f/(1.0f + exp(GELU_QUICK_COEF*x0)));
)"
R"(
)"
R"(        dst_row[i0] = gelu_quick*x1;
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_geglu_quick_f16(
)"
R"(    global char * src0,
)"
R"(    ulong  offset0,
)"
R"(    global char * src1,
)"
R"(    ulong  offset1,
)"
R"(    global char * dst,
)"
R"(    ulong  offsetd,
)"
R"(    ulong nb01,
)"
R"(    ulong nb11,
)"
R"(    int ne0,
)"
R"(    ulong nb1,
)"
R"(    int ne00_off,
)"
R"(    int ne10_off
)"
R"() {
)"
R"(    src0 = (global char*)((global char*)src0 + offset0);
)"
R"(    src1 = (global char*)((global char*)src1 + offset1);
)"
R"(    dst  = (global char*)((global char*)dst  + offsetd);
)"
R"(
)"
R"(    global half * src0_row = (global half *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
)"
R"(    global half * src1_row = (global half *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
)"
R"(    global half * dst_row  = (global half *) ((global char *) dst  + get_group_id(0)*nb1);
)"
R"(
)"
R"(    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
)"
R"(        const half x0 = src0_row[i0];
)"
R"(        const half x1 = src1_row[i0];
)"
R"(
)"
R"(        const half gelu_quick = x0*(1.0f/(1.0f + exp(GELU_QUICK_COEF*x0)));
)"
R"(
)"
R"(        dst_row[i0] = gelu_quick*x1;
)"
R"(    }
)"
R"(}
)"
