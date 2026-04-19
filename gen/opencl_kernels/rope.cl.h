R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// kernel_rope
)"
R"(//------------------------------------------------------------------------------
)"
R"(float rope_yarn_ramp(float low, float high, int i0) {
)"
R"(    const float y = (i0 / 2 - low) / max(0.001f, high - low);
)"
R"(    return 1.0f - min(1.0f, max(0.0f, y));
)"
R"(}
)"
R"(
)"
R"(// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
)"
R"(// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
)"
R"(float2 rope_yarn(
)"
R"(    float theta_extrap, float freq_scale, float2 corr_dims, int i0, float ext_factor, float mscale
)"
R"() {
)"
R"(    // Get n-d rotational scaling corrected for extrapolation
)"
R"(    float theta_interp = freq_scale * theta_extrap;
)"
R"(    float theta = theta_interp;
)"
R"(    if (ext_factor != 0.0f) {
)"
R"(        float ramp_mix = rope_yarn_ramp(corr_dims.s0, corr_dims.s1, i0) * ext_factor;
)"
R"(        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
)"
R"(
)"
R"(        // Get n-d magnitude scaling corrected for interpolation
)"
R"(        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
)"
R"(    }
)"
R"(    return (float2)(cos(theta) * mscale, sin(theta) * mscale);
)"
R"(}
)"
R"(
)"
R"(// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
)"
R"(// `corr_fac(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
)"
R"(float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
)"
R"(    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
)"
R"(}
)"
R"(
)"
R"(float2 rope_yarn_corr_dims(
)"
R"(    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow
)"
R"() {
)"
R"(    // start and end correction dims
)"
R"(    return (float2)(
)"
R"(        max(0.0f,         floor(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base))),
)"
R"(        min(n_dims - 1.0f, ceil(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)))
)"
R"(    );
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_rope_norm_f32(
)"
R"(        global void * src0,
)"
R"(        ulong offset0,
)"
R"(        global int * src1,
)"
R"(        ulong offset1,
)"
R"(        global float * src2,
)"
R"(        ulong offset2,
)"
R"(        global float * dst,
)"
R"(        ulong offsetd,
)"
R"(        int ne00,
)"
R"(        int ne01,
)"
R"(        int ne02,
)"
R"(        int ne03,
)"
R"(        ulong nb00,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        int ne0,
)"
R"(        int ne1,
)"
R"(        int ne2,
)"
R"(        int ne3,
)"
R"(        ulong nb0,
)"
R"(        ulong nb1,
)"
R"(        ulong nb2,
)"
R"(        ulong nb3,
)"
R"(        int n_past,
)"
R"(        int n_dims,
)"
R"(        int n_ctx_orig,
)"
R"(        float freq_base,
)"
R"(        float freq_scale,
)"
R"(        float ext_factor,
)"
R"(        float attn_factor,
)"
R"(        float beta_fast,
)"
R"(        float beta_slow
)"
R"() {
)"
R"(    src0 = (global void*)((global char*)src0 + offset0);
)"
R"(    src1 = (global int*)((global char*)src1 + offset1);
)"
R"(    src2 = (global float*)((global char*)src2 + offset2);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
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
R"(    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);
)"
R"(
)"
R"(    global int * pos = src1;
)"
R"(
)"
R"(    float theta_base = (float) pos[i2];
)"
R"(    float inv_ndims = -1.f/n_dims;
)"
R"(
)"
R"(    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
)"
R"(        if (i0 < n_dims) {
)"
R"(            int ic = i0/2;
)"
R"(
)"
R"(            float theta = theta_base * pow(freq_base, inv_ndims*i0);
)"
R"(
)"
R"(            float freq_factor = src2 != src0 ? src2[ic] : 1.0f;
)"
R"(
)"
R"(            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);
)"
R"(
)"
R"(            global float * src       = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
)"
R"(            global float * dst_data  = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
)"
R"(
)"
R"(            float x0 = src[0];
)"
R"(            float x1 = src[1];
)"
R"(
)"
R"(            dst_data[0] = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
)"
R"(            dst_data[1] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
)"
R"(        } else {
)"
R"(            global float * src      = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
)"
R"(            global float * dst_data = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
)"
R"(
)"
R"(            dst_data[0] = src[0];
)"
R"(            dst_data[1] = src[1];
)"
R"(        }
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_rope_norm_f16(
)"
R"(        global void * src0,
)"
R"(        ulong offset0,
)"
R"(        global int * src1,
)"
R"(        ulong offset1,
)"
R"(        global float * src2,
)"
R"(        ulong offset2,
)"
R"(        global float * dst,
)"
R"(        ulong offsetd,
)"
R"(        int ne00,
)"
R"(        int ne01,
)"
R"(        int ne02,
)"
R"(        int ne03,
)"
R"(        ulong nb00,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        int ne0,
)"
R"(        int ne1,
)"
R"(        int ne2,
)"
R"(        int ne3,
)"
R"(        ulong nb0,
)"
R"(        ulong nb1,
)"
R"(        ulong nb2,
)"
R"(        ulong nb3,
)"
R"(        int n_past,
)"
R"(        int n_dims,
)"
R"(        int n_ctx_orig,
)"
R"(        float freq_base,
)"
R"(        float freq_scale,
)"
R"(        float ext_factor,
)"
R"(        float attn_factor,
)"
R"(        float beta_fast,
)"
R"(        float beta_slow
)"
R"() {
)"
R"(    src0 = (global void*)((global char*)src0 + offset0);
)"
R"(    src1 = (global int*)((global char*)src1 + offset1);
)"
R"(    src2 = (global float*)((global char*)src2 + offset2);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
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
R"(    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);
)"
R"(
)"
R"(    global int * pos = src1;
)"
R"(
)"
R"(    float theta_base = (float) pos[i2];
)"
R"(    float inv_ndims = -1.f/n_dims;
)"
R"(
)"
R"(    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
)"
R"(        if (i0 < n_dims) {
)"
R"(            int ic = i0/2;
)"
R"(
)"
R"(            float theta = theta_base * pow(freq_base, inv_ndims*i0);
)"
R"(
)"
R"(            float freq_factor = src2 != src0 ? src2[ic] : 1.0f;
)"
R"(
)"
R"(            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);
)"
R"(
)"
R"(            global half * src       = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
)"
R"(            global half * dst_data  = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
)"
R"(
)"
R"(            float x0 = src[0];
)"
R"(            float x1 = src[1];
)"
R"(
)"
R"(            dst_data[0] = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
)"
R"(            dst_data[1] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
)"
R"(        } else {
)"
R"(            global half * src      = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
)"
R"(            global half * dst_data = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
)"
R"(
)"
R"(            dst_data[0] = src[0];
)"
R"(            dst_data[1] = src[1];
)"
R"(        }
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_rope_neox_f32(
)"
R"(        global void * src0,
)"
R"(        ulong offset0,
)"
R"(        global int * src1,
)"
R"(        ulong offset1,
)"
R"(        global float * src2,
)"
R"(        ulong offset2,
)"
R"(        global float * dst,
)"
R"(        ulong offsetd,
)"
R"(        int ne00,
)"
R"(        int ne01,
)"
R"(        int ne02,
)"
R"(        int ne03,
)"
R"(        ulong nb00,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        int ne0,
)"
R"(        int ne1,
)"
R"(        int ne2,
)"
R"(        int ne3,
)"
R"(        ulong nb0,
)"
R"(        ulong nb1,
)"
R"(        ulong nb2,
)"
R"(        ulong nb3,
)"
R"(        int n_past,
)"
R"(        int n_dims,
)"
R"(        int n_ctx_orig,
)"
R"(        float freq_base,
)"
R"(        float freq_scale,
)"
R"(        float ext_factor,
)"
R"(        float attn_factor,
)"
R"(        float beta_fast,
)"
R"(        float beta_slow
)"
R"() {
)"
R"(    src0 = (global void*)((global char*)src0 + offset0);
)"
R"(    src1 = (global int*)((global char*)src1 + offset1);
)"
R"(    src2 = (global float*)((global char*)src2 + offset2);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
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
R"(    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);
)"
R"(
)"
R"(    global int * pos = src1;
)"
R"(
)"
R"(    float theta_base = (float) pos[i2];
)"
R"(    float inv_ndims = -1.f/n_dims;
)"
R"(
)"
R"(    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
)"
R"(        if (i0 < n_dims) {
)"
R"(            int ic = i0/2;
)"
R"(
)"
R"(            const float theta = theta_base * pow(freq_base, inv_ndims*i0);
)"
R"(
)"
R"(            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;
)"
R"(
)"
R"(            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);
)"
R"(
)"
R"(            global float * src      = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
)"
R"(            global float * dst_data = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
)"
R"(
)"
R"(            const float x0 = src[0];
)"
R"(            const float x1 = src[n_dims/2];
)"
R"(
)"
R"(            dst_data[0]        = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
)"
R"(            dst_data[n_dims/2] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
)"
R"(        } else {
)"
R"(            global float * const src = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
)"
R"(            global float * dst_data  = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
)"
R"(
)"
R"(            dst_data[0] = src[0];
)"
R"(            dst_data[1] = src[1];
)"
R"(        }
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_rope_neox_f16(
)"
R"(        global void * src0,
)"
R"(        ulong offset0,
)"
R"(        global int * src1,
)"
R"(        ulong offset1,
)"
R"(        global float * src2,
)"
R"(        ulong offset2,
)"
R"(        global float * dst,
)"
R"(        ulong offsetd,
)"
R"(        int ne00,
)"
R"(        int ne01,
)"
R"(        int ne02,
)"
R"(        int ne03,
)"
R"(        ulong nb00,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        int ne0,
)"
R"(        int ne1,
)"
R"(        int ne2,
)"
R"(        int ne3,
)"
R"(        ulong nb0,
)"
R"(        ulong nb1,
)"
R"(        ulong nb2,
)"
R"(        ulong nb3,
)"
R"(        int n_past,
)"
R"(        int n_dims,
)"
R"(        int n_ctx_orig,
)"
R"(        float freq_base,
)"
R"(        float freq_scale,
)"
R"(        float ext_factor,
)"
R"(        float attn_factor,
)"
R"(        float beta_fast,
)"
R"(        float beta_slow
)"
R"() {
)"
R"(    src0 = (global void*)((global char*)src0 + offset0);
)"
R"(    src1 = (global int*)((global char*)src1 + offset1);
)"
R"(    src2 = (global float*)((global char*)src2 + offset2);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
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
R"(    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);
)"
R"(
)"
R"(    global int * pos = src1;
)"
R"(
)"
R"(    float theta_base = (float) pos[i2];
)"
R"(    float inv_ndims = -1.f/n_dims;
)"
R"(
)"
R"(    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
)"
R"(        if (i0 < n_dims) {
)"
R"(            int ic = i0/2;
)"
R"(
)"
R"(            const float theta = theta_base * pow(freq_base, inv_ndims*i0);
)"
R"(
)"
R"(            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;
)"
R"(
)"
R"(            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);
)"
R"(
)"
R"(            global half * src       = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
)"
R"(            global half * dst_data  = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
)"
R"(
)"
R"(            const float x0 = src[0];
)"
R"(            const float x1 = src[n_dims/2];
)"
R"(
)"
R"(            dst_data[0]        = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
)"
R"(            dst_data[n_dims/2] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
)"
R"(        } else {
)"
R"(            global half * const src = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
)"
R"(            global half * dst_data  = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
)"
R"(
)"
R"(            dst_data[0] = src[0];
)"
R"(            dst_data[1] = src[1];
)"
R"(        }
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_rope_multi_f32(
)"
R"(        global void * src0,
)"
R"(        ulong offset0,
)"
R"(        global int * src1,
)"
R"(        ulong offset1,
)"
R"(        global float * src2,
)"
R"(        ulong offset2,
)"
R"(        global float * dst,
)"
R"(        ulong offsetd,
)"
R"(        int ne00,
)"
R"(        int ne01,
)"
R"(        int ne02,
)"
R"(        int ne03,
)"
R"(        ulong nb00,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        int ne0,
)"
R"(        int ne1,
)"
R"(        int ne2,
)"
R"(        int ne3,
)"
R"(        ulong nb0,
)"
R"(        ulong nb1,
)"
R"(        ulong nb2,
)"
R"(        ulong nb3,
)"
R"(        int n_past,
)"
R"(        int n_dims,
)"
R"(        int n_ctx_orig,
)"
R"(        float freq_base,
)"
R"(        float freq_scale,
)"
R"(        float ext_factor,
)"
R"(        float attn_factor,
)"
R"(        float beta_fast,
)"
R"(        float beta_slow,
)"
R"(        int4 sections,
)"
R"(        int  is_imrope
)"
R"() {
)"
R"(    src0 = (global void*)((global char*)src0 + offset0);
)"
R"(    src1 = (global int*)((global char*)src1 + offset1);
)"
R"(    src2 = (global float*)((global char*)src2 + offset2);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
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
R"(    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);
)"
R"(
)"
R"(    global int * pos = src1;
)"
R"(
)"
R"(    const int sect_dims = sections.s0 + sections.s1 + sections.s2 + sections.s3;
)"
R"(    const int sec_w = sections.s1 + sections.s0;
)"
R"(
)"
R"(    float inv_ndims = -1.f/n_dims;
)"
R"(
)"
R"(    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
)"
R"(        if (i0 < n_dims) {
)"
R"(            int ic = i0/2;
)"
R"(
)"
R"(            const int sector = (i0 / 2) % sect_dims;
)"
R"(            float theta_base = 0.0f;
)"
R"(
)"
R"(            if (is_imrope) {
)"
R"(                if (sector % 3 == 1 && sector < 3 * sections.s1) { // h
)"
R"(                    theta_base = (float) pos[i2 + ne02 * 1];
)"
R"(                } else if (sector % 3 == 2 && sector < 3 * sections.s2) { // w
)"
R"(                    theta_base = (float) pos[i2 + ne02 * 2];
)"
R"(                } else if (sector % 3 == 0 && sector < 3 * sections.s0) { // t
)"
R"(                    theta_base = (float) pos[i2 + ne02 * 0];
)"
R"(                } else { // e
)"
R"(                    theta_base = (float) pos[i2 + ne02 * 3];
)"
R"(                }
)"
R"(            } else {
)"
R"(                if (sector < sections.s0) {
)"
R"(                    theta_base = pos[i2];
)"
R"(                }
)"
R"(                else if (sector >= sections.s0 && sector < sec_w) {
)"
R"(                    theta_base = pos[i2 + ne2 * 1];
)"
R"(                }
)"
R"(                else if (sector >= sec_w && sector < sec_w + sections.s2) {
)"
R"(                    theta_base = pos[i2 + ne2 * 2];
)"
R"(                }
)"
R"(                else if (sector >= sec_w + sections.s2) {
)"
R"(                    theta_base = pos[i2 + ne2 * 3];
)"
R"(                }
)"
R"(            }
)"
R"(
)"
R"(            const float theta = theta_base * pow(freq_base, inv_ndims*i0);
)"
R"(
)"
R"(            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;
)"
R"(
)"
R"(            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);
)"
R"(
)"
R"(            global float * src      = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
)"
R"(            global float * dst_data = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
)"
R"(
)"
R"(            const float x0 = src[0];
)"
R"(            const float x1 = src[n_dims/2];
)"
R"(
)"
R"(            dst_data[0]        = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
)"
R"(            dst_data[n_dims/2] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
)"
R"(        } else {
)"
R"(            global float * const src = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
)"
R"(            global float * dst_data  = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
)"
R"(
)"
R"(            dst_data[0] = src[0];
)"
R"(            dst_data[1] = src[1];
)"
R"(        }
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_rope_multi_f16(
)"
R"(        global void * src0,
)"
R"(        ulong offset0,
)"
R"(        global int * src1,
)"
R"(        ulong offset1,
)"
R"(        global float * src2,
)"
R"(        ulong offset2,
)"
R"(        global half * dst,
)"
R"(        ulong offsetd,
)"
R"(        int ne00,
)"
R"(        int ne01,
)"
R"(        int ne02,
)"
R"(        int ne03,
)"
R"(        ulong nb00,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        int ne0,
)"
R"(        int ne1,
)"
R"(        int ne2,
)"
R"(        int ne3,
)"
R"(        ulong nb0,
)"
R"(        ulong nb1,
)"
R"(        ulong nb2,
)"
R"(        ulong nb3,
)"
R"(        int n_past,
)"
R"(        int n_dims,
)"
R"(        int n_ctx_orig,
)"
R"(        float freq_base,
)"
R"(        float freq_scale,
)"
R"(        float ext_factor,
)"
R"(        float attn_factor,
)"
R"(        float beta_fast,
)"
R"(        float beta_slow,
)"
R"(        int4 sections,
)"
R"(        int  is_imrope
)"
R"() {
)"
R"(    src0 = (global void*)((global char*)src0 + offset0);
)"
R"(    src1 = (global int*)((global char*)src1 + offset1);
)"
R"(    src2 = (global float*)((global char*)src2 + offset2);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
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
R"(    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);
)"
R"(
)"
R"(    global int * pos = src1;
)"
R"(
)"
R"(    const int sect_dims = sections.s0 + sections.s1 + sections.s2 + sections.s3;
)"
R"(    const int sec_w = sections.s1 + sections.s0;
)"
R"(
)"
R"(    float inv_ndims = -1.f/n_dims;
)"
R"(
)"
R"(    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
)"
R"(        if (i0 < n_dims) {
)"
R"(            int ic = i0/2;
)"
R"(
)"
R"(            const int sector = (i0 / 2) % sect_dims;
)"
R"(            float theta_base = 0.0f;
)"
R"(
)"
R"(            if (is_imrope) {
)"
R"(                if (sector % 3 == 1 && sector < 3 * sections.s1) { // h
)"
R"(                    theta_base = (float) pos[i2 + ne02 * 1];
)"
R"(                } else if (sector % 3 == 2 && sector < 3 * sections.s2) { // w
)"
R"(                    theta_base = (float) pos[i2 + ne02 * 2];
)"
R"(                } else if (sector % 3 == 0 && sector < 3 * sections.s0) { // t
)"
R"(                    theta_base = (float) pos[i2 + ne02 * 0];
)"
R"(                } else { // e
)"
R"(                    theta_base = (float) pos[i2 + ne02 * 3];
)"
R"(                }
)"
R"(            } else {
)"
R"(                if (sector < sections.s0) {
)"
R"(                    theta_base = pos[i2];
)"
R"(                }
)"
R"(                else if (sector >= sections.s0 && sector < sec_w) {
)"
R"(                    theta_base = pos[i2 + ne2 * 1];
)"
R"(                }
)"
R"(                else if (sector >= sec_w && sector < sec_w + sections.s2) {
)"
R"(                    theta_base = pos[i2 + ne2 * 2];
)"
R"(                }
)"
R"(                else if (sector >= sec_w + sections.s2) {
)"
R"(                    theta_base = pos[i2 + ne2 * 3];
)"
R"(                }
)"
R"(            }
)"
R"(
)"
R"(            const float theta = theta_base * pow(freq_base, inv_ndims*i0);
)"
R"(
)"
R"(            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;
)"
R"(
)"
R"(            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);
)"
R"(
)"
R"(            global half * src      = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
)"
R"(            global half * dst_data = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
)"
R"(
)"
R"(            const float x0 = src[0];
)"
R"(            const float x1 = src[n_dims/2];
)"
R"(
)"
R"(            dst_data[0]        = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
)"
R"(            dst_data[n_dims/2] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
)"
R"(        } else {
)"
R"(            global half * const src = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
)"
R"(            global half * dst_data  = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
)"
R"(
)"
R"(            dst_data[0] = src[0];
)"
R"(            dst_data[1] = src[1];
)"
R"(        }
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_rope_vision_f32(
)"
R"(        global void * src0,
)"
R"(        ulong offset0,
)"
R"(        global int * src1,
)"
R"(        ulong offset1,
)"
R"(        global float * src2,
)"
R"(        ulong offset2,
)"
R"(        global float * dst,
)"
R"(        ulong offsetd,
)"
R"(        int ne00,
)"
R"(        int ne01,
)"
R"(        int ne02,
)"
R"(        int ne03,
)"
R"(        ulong nb00,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        int ne0,
)"
R"(        int ne1,
)"
R"(        int ne2,
)"
R"(        int ne3,
)"
R"(        ulong nb0,
)"
R"(        ulong nb1,
)"
R"(        ulong nb2,
)"
R"(        ulong nb3,
)"
R"(        int n_past,
)"
R"(        int n_dims,
)"
R"(        int n_ctx_orig,
)"
R"(        float freq_base,
)"
R"(        float freq_scale,
)"
R"(        float ext_factor,
)"
R"(        float attn_factor,
)"
R"(        float beta_fast,
)"
R"(        float beta_slow,
)"
R"(        int4 sections
)"
R"() {
)"
R"(    src0 = (global void*)((global char*)src0 + offset0);
)"
R"(    src1 = (global int*)((global char*)src1 + offset1);
)"
R"(    src2 = (global float*)((global char*)src2 + offset2);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
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
R"(    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);
)"
R"(
)"
R"(    global int * pos = src1;
)"
R"(
)"
R"(    const int sect_dims = sections.s0 + sections.s1;
)"
R"(    const int sec_w = sections.s1 + sections.s0;
)"
R"(
)"
R"(    float inv_ndims = -1.f/n_dims;
)"
R"(
)"
R"(    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
)"
R"(        int ic = i0/2;
)"
R"(
)"
R"(        const int sector = (i0/2) % sect_dims;
)"
R"(        float theta_base = 0.0f;
)"
R"(
)"
R"(        if (sector < sections.s0) {
)"
R"(            const int p = sector;
)"
R"(            theta_base = pos[i2] * pow(freq_base, inv_ndims*2.0f*p);
)"
R"(        } else if (sector >= sections.s0 && sector < sec_w) {
)"
R"(            const int p = sector - sections.s0;
)"
R"(            theta_base = pos[i2 + ne2] * pow(freq_base, inv_ndims*2.0f*p);
)"
R"(        }
)"
R"(
)"
R"(        const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;
)"
R"(
)"
R"(        float2 cos_sin_theta = rope_yarn(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);
)"
R"(
)"
R"(        global float * src      = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
)"
R"(        global float * dst_data = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
)"
R"(
)"
R"(        const float x0 = src[0];
)"
R"(        const float x1 = src[n_dims];
)"
R"(
)"
R"(        dst_data[0]      = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
)"
R"(        dst_data[n_dims] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_rope_vision_f16(
)"
R"(        global void * src0,
)"
R"(        ulong offset0,
)"
R"(        global int * src1,
)"
R"(        ulong offset1,
)"
R"(        global float * src2,
)"
R"(        ulong offset2,
)"
R"(        global half * dst,
)"
R"(        ulong offsetd,
)"
R"(        int ne00,
)"
R"(        int ne01,
)"
R"(        int ne02,
)"
R"(        int ne03,
)"
R"(        ulong nb00,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        int ne0,
)"
R"(        int ne1,
)"
R"(        int ne2,
)"
R"(        int ne3,
)"
R"(        ulong nb0,
)"
R"(        ulong nb1,
)"
R"(        ulong nb2,
)"
R"(        ulong nb3,
)"
R"(        int n_past,
)"
R"(        int n_dims,
)"
R"(        int n_ctx_orig,
)"
R"(        float freq_base,
)"
R"(        float freq_scale,
)"
R"(        float ext_factor,
)"
R"(        float attn_factor,
)"
R"(        float beta_fast,
)"
R"(        float beta_slow,
)"
R"(        int4 sections
)"
R"() {
)"
R"(    src0 = (global void*)((global char*)src0 + offset0);
)"
R"(    src1 = (global int*)((global char*)src1 + offset1);
)"
R"(    src2 = (global float*)((global char*)src2 + offset2);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
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
R"(    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);
)"
R"(
)"
R"(    global int * pos = src1;
)"
R"(
)"
R"(    const int sect_dims = sections.s0 + sections.s1;
)"
R"(    const int sec_w = sections.s1 + sections.s0;
)"
R"(
)"
R"(    float inv_ndims = -1.f/n_dims;
)"
R"(
)"
R"(    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
)"
R"(        int ic = i0/2;
)"
R"(
)"
R"(        const int sector = (i0/2) % sect_dims;
)"
R"(        float theta_base = 0.0f;
)"
R"(
)"
R"(        if (sector < sections.s0) {
)"
R"(            const int p = sector;
)"
R"(            theta_base = pos[i2] * pow(freq_base, inv_ndims*2.0f*p);
)"
R"(        } else if (sector >= sections.s0 && sector < sec_w) {
)"
R"(            const int p = sector - sections.s0;
)"
R"(            theta_base = pos[i2 + ne2] * pow(freq_base, inv_ndims*2.0f*p);
)"
R"(        }
)"
R"(
)"
R"(        const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;
)"
R"(
)"
R"(        float2 cos_sin_theta = rope_yarn(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);
)"
R"(
)"
R"(        global half * src      = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
)"
R"(        global half * dst_data = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
)"
R"(
)"
R"(        const float x0 = src[0];
)"
R"(        const float x1 = src[n_dims];
)"
R"(
)"
R"(        dst_data[0]      = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
)"
R"(        dst_data[n_dims] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
)"
R"(    }
)"
R"(}
)"
