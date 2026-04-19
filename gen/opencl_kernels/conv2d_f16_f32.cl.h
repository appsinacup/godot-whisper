R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(
)"
R"(#if defined(cl_qcom_reqd_sub_group_size)
)"
R"(#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
)"
R"(#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
)"
R"(#else
)"
R"(#define REQD_SUBGROUP_SIZE_128
)"
R"(#endif
)"
R"(
)"
R"(#define T_ACCUM float4
)"
R"(#define VEC_SIZE 4
)"
R"(
)"
R"(#define BS_K 64
)"
R"(#define BS_NPQ 64
)"
R"(#define BS_CRS 16
)"
R"(
)"
R"(#define TS_K 4
)"
R"(#define TS_NPQ 8
)"
R"(
)"
R"(#define WG_K (BS_K / TS_K)
)"
R"(#define WG_NPQ (BS_NPQ / TS_NPQ)
)"
R"(
)"
R"(#define BS_NPQ_VEC (BS_NPQ / VEC_SIZE)
)"
R"(#define TS_NPQ_VEC (TS_NPQ / VEC_SIZE)
)"
R"(
)"
R"(static inline uint splitWork(uint work_size, uint block_size){
)"
R"(    return (work_size + block_size - 1) / block_size;
)"
R"(}
)"
R"(
)"
R"(REQD_SUBGROUP_SIZE_128
)"
R"(kernel void kernel_conv_2d(
)"
R"(    global void* p_knl,
)"
R"(    ulong off_knl,
)"
R"(    global void* p_src,
)"
R"(    ulong off_src,
)"
R"(    global void* p_dst,
)"
R"(    ulong off_dst,
)"
R"(    local void* shared,
)"
R"(    uint Cout, uint Cin, uint N,
)"
R"(    uint KW, uint KH, uint W, uint H, uint OW, uint OH,
)"
R"(    uint s0, uint s1, uint p0, uint p1, uint d0, uint d1,
)"
R"(    uint nb01, uint nb02, uint nb03,
)"
R"(    uint nb11, uint nb12, uint nb13,
)"
R"(    uint nb1, uint nb2, uint nb3
)"
R"() {
)"
R"(    global half* knl_data = (global half*) ((global char*)p_knl + off_knl);
)"
R"(    global float* src_data = (global float*) ((global char*)p_src + off_src);
)"
R"(    global float* dst_data = (global float*) ((global char*)p_dst + off_dst);
)"
R"(
)"
R"(    const uint K = Cout;
)"
R"(    const uint CRS = Cin*KH*KW;
)"
R"(    const uint NPQ = N*OH*OW;
)"
R"(
)"
R"(    const uint lid_k = get_local_id(0);
)"
R"(    const uint lid_npq = get_local_id(1);
)"
R"(    const uint tid = lid_npq * WG_K + lid_k;
)"
R"(
)"
R"(    const uint B_idx_K = get_group_id(0);
)"
R"(    const uint B_idx_NPQ = get_group_id(1);
)"
R"(
)"
R"(    const uint offset_k = B_idx_K * BS_K;
)"
R"(    const uint offset_npq = B_idx_NPQ * BS_NPQ;
)"
R"(
)"
R"(    local half* Ash = (local half*)shared;
)"
R"(    local float4* Bsh = (local float4*) &Ash[BS_K * BS_CRS];
)"
R"(
)"
R"(    T_ACCUM regC[TS_K][TS_NPQ_VEC];
)"
R"(    for (int i = 0; i < TS_K; ++i) {
)"
R"(        for (int j = 0; j < TS_NPQ_VEC; ++j) {
)"
R"(            regC[i][j] = (T_ACCUM)(0.0f);
)"
R"(        }
)"
R"(    }
)"
R"(
)"
R"(    const uint NB_CRS = splitWork(CRS, BS_CRS);
)"
R"(
)"
R"(    for (uint B_idx_CRS = 0; B_idx_CRS < NB_CRS; ++B_idx_CRS) {
)"
R"(        const uint offset_crs = B_idx_CRS * BS_CRS;
)"
R"(
)"
R"(        for (int i = tid; i < BS_K * BS_CRS; i += (WG_K * WG_NPQ)) {
)"
R"(            const uint k_l = i / BS_CRS;
)"
R"(            const uint crs_l = i % BS_CRS;
)"
R"(            const uint k_g = offset_k + k_l;
)"
R"(            const uint crs_g = offset_crs + crs_l;
)"
R"(
)"
R"(            if (k_g < K && crs_g < CRS) {
)"
R"(                const uint Cin_idx = crs_g / (KW*KH);
)"
R"(                const uint KH_idx = (crs_g - Cin_idx*KW*KH) / KW;
)"
R"(                const uint KW_idx = crs_g - Cin_idx*KW*KH - KH_idx*KW;
)"
R"(                const uint knl_idx = KW_idx + KH_idx*nb01 + Cin_idx*nb02 + k_g*nb03;
)"
R"(                Ash[k_l * BS_CRS + crs_l] = knl_data[knl_idx];
)"
R"(            } else {
)"
R"(                Ash[k_l * BS_CRS + crs_l] = (half)0.0f;
)"
R"(            }
)"
R"(        }
)"
R"(
)"
R"(        for (int i = tid; i < BS_CRS * BS_NPQ_VEC; i += (WG_K * WG_NPQ)) {
)"
R"(            const uint crs_l = i / BS_NPQ_VEC;
)"
R"(            const uint npq_l_vec = i % BS_NPQ_VEC;
)"
R"(            const uint crs_g = offset_crs + crs_l;
)"
R"(
)"
R"(            float4 val = (float4)(0.0f);
)"
R"(            if (crs_g < CRS) {
)"
R"(                const uint Cin_idx = crs_g / (KW * KH);
)"
R"(                const uint KH_idx = (crs_g - Cin_idx * KW * KH) / KW;
)"
R"(                const uint KW_idx = crs_g - Cin_idx * KW * KH - KH_idx * KW;
)"
R"(                for (int v = 0; v < VEC_SIZE; ++v) {
)"
R"(                    const uint npq_g = offset_npq + npq_l_vec * VEC_SIZE + v;
)"
R"(                    if (npq_g < NPQ) {
)"
R"(                        const uint N_idx = npq_g / (OH * OW);
)"
R"(                        const uint pq_idx = npq_g % (OH * OW);
)"
R"(                        const uint OH_idx = pq_idx / OW;
)"
R"(                        const uint OW_idx = pq_idx % OW;
)"
R"(                        const int H_idx = (int)(OH_idx * s1 + KH_idx * d1 - p1);
)"
R"(                        const int W_idx = (int)(OW_idx * s0 + KW_idx * d0 - p0);
)"
R"(
)"
R"(                        if (H_idx >= 0 && H_idx < H && W_idx >= 0 && W_idx < W) {
)"
R"(                            const uint src_idx = W_idx + H_idx * nb11 + Cin_idx * nb12 + N_idx * nb13;
)"
R"(                            ((float*)&val)[v] = src_data[src_idx];
)"
R"(                        }
)"
R"(                    }
)"
R"(                }
)"
R"(            }
)"
R"(            Bsh[crs_l * BS_NPQ_VEC + npq_l_vec] = val;
)"
R"(        }
)"
R"(
)"
R"(        barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(
)"
R"(        #pragma unroll
)"
R"(        for (uint crs_l = 0; crs_l < BS_CRS; ++crs_l) {
)"
R"(            half regA[TS_K];
)"
R"(            for (uint k_l_reg = 0; k_l_reg < TS_K; ++k_l_reg) {
)"
R"(                regA[k_l_reg] = Ash[(lid_k * TS_K + k_l_reg) * BS_CRS + crs_l];
)"
R"(            }
)"
R"(
)"
R"(            for (uint npq_l_vec_reg = 0; npq_l_vec_reg < TS_NPQ_VEC; ++npq_l_vec_reg) {
)"
R"(                float4 regB = Bsh[crs_l * BS_NPQ_VEC + lid_npq * TS_NPQ_VEC + npq_l_vec_reg];
)"
R"(                for (uint k_l_reg = 0; k_l_reg < TS_K; ++k_l_reg) {
)"
R"(                    regC[k_l_reg][npq_l_vec_reg] = mad(convert_float(regA[k_l_reg]), regB, regC[k_l_reg][npq_l_vec_reg]);
)"
R"(                }
)"
R"(            }
)"
R"(        }
)"
R"(        barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(    }
)"
R"(
)"
R"(    for (uint k_l_reg = 0; k_l_reg < TS_K; ++k_l_reg) {
)"
R"(        const uint k_g = offset_k + lid_k * TS_K + k_l_reg;
)"
R"(        if (k_g >= K) continue;
)"
R"(
)"
R"(        for (uint npq_l_vec_reg = 0; npq_l_vec_reg < TS_NPQ_VEC; ++npq_l_vec_reg) {
)"
R"(            const uint npq_g_base = offset_npq + (lid_npq * TS_NPQ_VEC + npq_l_vec_reg) * VEC_SIZE;
)"
R"(
)"
R"(            const uint N_idx = npq_g_base / (OH * OW);
)"
R"(            const uint pq_idx = npq_g_base % (OH * OW);
)"
R"(            const uint OH_idx = pq_idx / OW;
)"
R"(            const uint OW_idx = pq_idx % OW;
)"
R"(
)"
R"(            if (nb1 == OW && OW_idx + VEC_SIZE <= OW && npq_g_base + VEC_SIZE <= NPQ) {
)"
R"(                const uint dst_idx = OW_idx + OH_idx*nb1 + k_g*nb2 + N_idx*nb3;
)"
R"(                vstore4(regC[k_l_reg][npq_l_vec_reg], 0, &dst_data[dst_idx]);
)"
R"(            } else {
)"
R"(                T_ACCUM res = regC[k_l_reg][npq_l_vec_reg];
)"
R"(                for (int v = 0; v < VEC_SIZE; ++v) {
)"
R"(                    const uint npq_g = npq_g_base + v;
)"
R"(                    if (npq_g < NPQ) {
)"
R"(                        const uint N_idx_s = npq_g / (OH*OW);
)"
R"(                        const uint pq_idx_s = npq_g % (OH*OW);
)"
R"(                        const uint OH_idx_s = pq_idx_s / OW;
)"
R"(                        const uint OW_idx_s = pq_idx_s % OW;
)"
R"(                        const uint dst_idx_s = OW_idx_s + OH_idx_s*nb1 + k_g*nb2 + N_idx_s*nb3;
)"
R"(                        dst_data[dst_idx_s] = ((float*)&res)[v];
)"
R"(                    }
)"
R"(                }
)"
R"(            }
)"
R"(        }
)"
R"(    }
)"
R"(}
)"
