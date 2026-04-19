R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
		R"(
)"
		R"(#define ACC_TYPE float
)"
		R"(#define ACC_TYPE4 float4
)"
		R"(#define DATA_TYPE float
)"
		R"(#define DATA_TYPE4 float4
)"
		R"(#define MASK_DATA_TYPE half
)"
		R"(#define CONVERT_ACC4(x) (x)
)"
		R"(#define CONVERT_DATA4(x) (x)
)"
		R"(
)"
		R"(#define DK_VEC (DK/4)
)"
		R"(#define DV_VEC (DV/4)
)"
		R"(#define WG_SIZE (BLOCK_M)
)"
		R"(#define Q1_WG_SIZE 64
)"
		R"(
)"
		R"(inline float get_alibi_slope(
)"
		R"(    const float max_bias, const uint h, const uint n_head_log2, const float m0, const float m1
)"
		R"() {
)"
		R"(    if (max_bias <= 0.0f) {
)"
		R"(        return 1.0f;
)"
		R"(    }
)"
		R"(    const float base = h < n_head_log2 ? m0 : m1;
)"
		R"(    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;
)"
		R"(
)"
		R"(    return pow(base, exph);
)"
		R"(}
)"
		R"(__kernel void flash_attn_f32(
)"
		R"(    const global void * q_void, ulong q_offset,
)"
		R"(    const global void * k_void, ulong k_offset,
)"
		R"(    const global void * v_void, ulong v_offset,
)"
		R"(    global void * o_void, ulong o_offset,
)"
		R"(    const float scale,
)"
		R"(    const int n_q,
)"
		R"(    const int n_kv,
)"
		R"(    const int is_causal,
)"
		R"(    const int n_head,
)"
		R"(    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
)"
		R"(    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
)"
		R"(    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
)"
		R"(    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
)"
		R"(    const float max_bias,
)"
		R"(    const float m0,
)"
		R"(    const float m1,
)"
		R"(    const int n_head_log2,
)"
		R"(    const float logit_softcap,
)"
		R"(    const int n_head_kv,
)"
		R"(    const global void* mask_void,
)"
		R"(    const ulong mask_offset,
)"
		R"(    const ulong mask_nb1,
)"
		R"(    const ulong mask_nb2,
)"
		R"(    const ulong mask_nb3,
)"
		R"(    const int mask_ne2,
)"
		R"(    const int mask_ne3,
)"
		R"(    const global void* sinks_void,
)"
		R"(    const ulong sinks_offset
)"
		R"() {
)"
		R"(    const int tid = get_local_id(0);
)"
		R"(    const int block_q_idx = get_group_id(0);
)"
		R"(    const int head_batch_idx = get_global_id(1);
)"
		R"(
)"
		R"(    const int my_query_row = block_q_idx * BLOCK_M + tid;
)"
		R"(
)"
		R"(    const int batch_idx = head_batch_idx / n_head;
)"
		R"(    const int head_idx = head_batch_idx % n_head;
)"
		R"(
)"
		R"(    const int gqa_ratio = n_head / n_head_kv;
)"
		R"(    const int head_kv_idx = head_idx / gqa_ratio;
)"
		R"(
)"
		R"(    const global char* q_base = (const global char*)q_void + q_offset;
)"
		R"(    const global char* k_base = (const global char*)k_void + k_offset;
)"
		R"(    const global char* v_base = (const global char*)v_void + v_offset;
)"
		R"(    global char* o_base = (global char*)o_void + o_offset;
)"
		R"(
)"
		R"(    const global char* mask_base = NULL;
)"
		R"(    if (mask_void != NULL) {
)"
		R"(        const int mask_head_idx = head_idx % mask_ne2;
)"
		R"(        const int mask_batch_idx = batch_idx % mask_ne3;
)"
		R"(        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
)"
		R"(    }
)"
		R"(
)"
		R"(    ACC_TYPE4 q_priv[DK_VEC];
)"
		R"(    if (my_query_row < n_q) {
)"
		R"(        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + my_query_row * q_nb1;
)"
		R"(        const global DATA_TYPE4* q_ptr = (const global DATA_TYPE4*)(q_base + q_row_offset);
)"
		R"(        #pragma unroll
)"
		R"(        for (int i = 0; i < DK_VEC; ++i) {
)"
		R"(            q_priv[i] = CONVERT_ACC4(q_ptr[i]);
)"
		R"(        }
)"
		R"(    }
)"
		R"(
)"
		R"(    ACC_TYPE4 o_acc[DV_VEC];
)"
		R"(    #pragma unroll
)"
		R"(    for (int i = 0; i < DV_VEC; ++i) {
)"
		R"(        o_acc[i] = (ACC_TYPE4)(0.0f);
)"
		R"(    }
)"
		R"(    ACC_TYPE m_i = -INFINITY;
)"
		R"(    ACC_TYPE l_i = 0.0f;
)"
		R"(
)"
		R"(    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);
)"
		R"(
)"
		R"(    __local DATA_TYPE4 l_k[BLOCK_N][DK_VEC];
)"
		R"(    __local DATA_TYPE4 l_v[BLOCK_N][DV_VEC];
)"
		R"(
)"
		R"(    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
)"
		R"(        for (int i = tid; i < BLOCK_N * DK_VEC; i += WG_SIZE) {
)"
		R"(            const int row = i / DK_VEC;
)"
		R"(            const int col = i % DK_VEC;
)"
		R"(            const int k_row_idx = k_start + row;
)"
		R"(            if (k_row_idx < n_kv) {
)"
		R"(                const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_row_idx * k_nb1;
)"
		R"(                l_k[row][col] = ((__global DATA_TYPE4*)(k_base + k_row_offset))[col];
)"
		R"(            }
)"
		R"(        }
)"
		R"(        for (int i = tid; i < BLOCK_N * DV_VEC; i += WG_SIZE) {
)"
		R"(            const int row = i / DV_VEC;
)"
		R"(            const int col = i % DV_VEC;
)"
		R"(            const int v_row_idx = k_start + row;
)"
		R"(            if (v_row_idx < n_kv) {
)"
		R"(                const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + v_row_idx * v_nb1;
)"
		R"(                l_v[row][col] = ((__global DATA_TYPE4*)(v_base + v_row_offset))[col];
)"
		R"(            }
)"
		R"(        }
)"
		R"(        barrier(CLK_LOCAL_MEM_FENCE);
)"
		R"(
)"
		R"(        if (my_query_row >= n_q) {
)"
		R"(            continue;
)"
		R"(        }
)"
		R"(
)"
		R"(        for (int j = 0; j < BLOCK_N; j += 2) {
)"
		R"(            const int k_row0 = k_start + j;
)"
		R"(            const int k_row1 = k_start + j + 1;
)"
		R"(
)"
		R"(            ACC_TYPE4 dot_acc0 = (ACC_TYPE4)(0.0f);
)"
		R"(            ACC_TYPE4 dot_acc1 = (ACC_TYPE4)(0.0f);
)"
		R"(            #pragma unroll
)"
		R"(            for (int k = 0; k < DK_VEC; k++) {
)"
		R"(                dot_acc0 = mad(q_priv[k], CONVERT_ACC4(l_k[j][k]), dot_acc0);
)"
		R"(                dot_acc1 = mad(q_priv[k], CONVERT_ACC4(l_k[j+1][k]), dot_acc1);
)"
		R"(            }
)"
		R"(            ACC_TYPE score0 = (dot_acc0.s0 + dot_acc0.s1 + dot_acc0.s2 + dot_acc0.s3) * scale;
)"
		R"(            ACC_TYPE score1 = (dot_acc1.s0 + dot_acc1.s1 + dot_acc1.s2 + dot_acc1.s3) * scale;
)"
		R"(
)"
		R"(            if (is_causal) {
)"
		R"(                if (k_row0 > (n_kv - n_q + my_query_row)) score0 = -INFINITY;
)"
		R"(                if (k_row1 > (n_kv - n_q + my_query_row)) score1 = -INFINITY;
)"
		R"(            }
)"
		R"(
)"
		R"(            if (k_row0 >= n_kv) score0 = -INFINITY;
)"
		R"(            if (k_row1 >= n_kv) score1 = -INFINITY;
)"
		R"(
)"
		R"(            if (mask_base != NULL) {
)"
		R"(                const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base + my_query_row * mask_nb1);
)"
		R"(                if (k_row0 < n_kv) score0 += slope * (ACC_TYPE)mask_ptr[k_row0];
)"
		R"(                if (k_row1 < n_kv) score1 += slope * (ACC_TYPE)mask_ptr[k_row1];
)"
		R"(            }
)"
		R"(
)"
		R"(            if (logit_softcap > 0.0f) {
)"
		R"(                score0 = logit_softcap * tanh(score0 / logit_softcap);
)"
		R"(                score1 = logit_softcap * tanh(score1 / logit_softcap);
)"
		R"(            }
)"
		R"(
)"
		R"(            const ACC_TYPE m_new = max(m_i, max(score0, score1));
)"
		R"(            const ACC_TYPE p0 = exp(score0 - m_new);
)"
		R"(            const ACC_TYPE p1 = exp(score1 - m_new);
)"
		R"(            const ACC_TYPE scale_prev = exp(m_i - m_new);
)"
		R"(
)"
		R"(            #pragma unroll
)"
		R"(            for (int i = 0; i < DV_VEC; ++i) {
)"
		R"(                o_acc[i] = o_acc[i] * scale_prev + p0 * CONVERT_ACC4(l_v[j][i]) + p1 * CONVERT_ACC4(l_v[j+1][i]);
)"
		R"(            }
)"
		R"(            l_i = l_i * scale_prev + p0 + p1;
)"
		R"(            m_i = m_new;
)"
		R"(        }
)"
		R"(    }
)"
		R"(
)"
		R"(    if (my_query_row < n_q) {
)"
		R"(        if (sinks_void != NULL) {
)"
		R"(            const global ACC_TYPE* sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
)"
		R"(            const ACC_TYPE m_sink = sinks_ptr[head_idx];
)"
		R"(            const ACC_TYPE m_final = max(m_i, m_sink);
)"
		R"(
)"
		R"(            const ACC_TYPE scale_o = exp(m_i - m_final);
)"
		R"(            #pragma unroll
)"
		R"(            for (int i = 0; i < DV_VEC; ++i) {
)"
		R"(                o_acc[i] *= scale_o;
)"
		R"(            }
)"
		R"(
)"
		R"(            l_i = l_i * exp(m_i - m_final) + exp(m_sink - m_final);
)"
		R"(        }
)"
		R"(
)"
		R"(        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
)"
		R"(        global DATA_TYPE4 *o_row = (global DATA_TYPE4 *)(o_base + o_row_offset);
)"
		R"(        if (l_i > 0.0f) {
)"
		R"(            const ACC_TYPE l_inv = 1.0f / l_i;
)"
		R"(            #pragma unroll
)"
		R"(            for (int i = 0; i < DV_VEC; ++i) {
)"
		R"(                o_row[i] = CONVERT_DATA4(o_acc[i] * l_inv);
)"
		R"(            }
)"
		R"(        } else {
)"
		R"(            #pragma unroll
)"
		R"(            for (int i = 0; i < DV_VEC; ++i) {
)"
		R"(                o_row[i] = (DATA_TYPE4)(0.0f);
)"
		R"(            }
)"
		R"(        }
)"
		R"(    }
)"
		R"(}
)"
		R"(
)"
		R"(__kernel void flash_attn_f32_q1(
)"
		R"(    const global void * q_void, ulong q_offset,
)"
		R"(    const global void * k_void, ulong k_offset,
)"
		R"(    const global void * v_void, ulong v_offset,
)"
		R"(    global void * o_void, ulong o_offset,
)"
		R"(    const float scale,
)"
		R"(    const int n_q,
)"
		R"(    const int n_kv,
)"
		R"(    const int is_causal,
)"
		R"(    const int n_head,
)"
		R"(    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
)"
		R"(    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
)"
		R"(    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
)"
		R"(    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
)"
		R"(    const float max_bias,
)"
		R"(    const float m0,
)"
		R"(    const float m1,
)"
		R"(    const int n_head_log2,
)"
		R"(    const float logit_softcap,
)"
		R"(    const int n_head_kv,
)"
		R"(    const global void* mask_void,
)"
		R"(    const ulong mask_offset,
)"
		R"(    const ulong mask_nb1,
)"
		R"(    const ulong mask_nb2,
)"
		R"(    const ulong mask_nb3,
)"
		R"(    const int mask_ne2,
)"
		R"(    const int mask_ne3,
)"
		R"(    const global void* sinks_void,
)"
		R"(    const ulong sinks_offset
)"
		R"() {
)"
		R"(    const int tid = get_local_id(0);
)"
		R"(    const int head_batch_idx = get_global_id(1);
)"
		R"(
)"
		R"(    const int batch_idx = head_batch_idx / n_head;
)"
		R"(    const int head_idx = head_batch_idx % n_head;
)"
		R"(
)"
		R"(    const int gqa_ratio = n_head / n_head_kv;
)"
		R"(    const int head_kv_idx = head_idx / gqa_ratio;
)"
		R"(
)"
		R"(    const global char* q_base = (const global char*)q_void + q_offset;
)"
		R"(    const global char* k_base = (const global char*)k_void + k_offset;
)"
		R"(    const global char* v_base = (const global char*)v_void + v_offset;
)"
		R"(    global char* o_base = (global char*)o_void + o_offset;
)"
		R"(
)"
		R"(    const global char* mask_base = NULL;
)"
		R"(    if (mask_void != NULL) {
)"
		R"(        const int mask_head_idx = head_idx % mask_ne2;
)"
		R"(        const int mask_batch_idx = batch_idx % mask_ne3;
)"
		R"(        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
)"
		R"(    }
)"
		R"(
)"
		R"(    ACC_TYPE4 q_priv[DK_VEC];
)"
		R"(    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2;
)"
		R"(    const global DATA_TYPE4* q_ptr = (const global DATA_TYPE4*)(q_base + q_row_offset);
)"
		R"(    #pragma unroll
)"
		R"(    for (int i = 0; i < DK_VEC; ++i) {
)"
		R"(        q_priv[i] = CONVERT_ACC4(q_ptr[i]);
)"
		R"(    }
)"
		R"(
)"
		R"(    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);
)"
		R"(
)"
		R"(    const global ACC_TYPE* sinks_ptr = NULL;
)"
		R"(    if (sinks_void != NULL) {
)"
		R"(        sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
)"
		R"(    }
)"
		R"(
)"
		R"(    ACC_TYPE m_i = (sinks_ptr != NULL) ? sinks_ptr[head_idx] : -INFINITY;
)"
		R"(    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
)"
		R"(        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
)"
		R"(        const global DATA_TYPE4* k_ptr = (const global DATA_TYPE4*)(k_base + k_row_offset);
)"
		R"(        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
)"
		R"(        #pragma unroll
)"
		R"(        for (int k = 0; k < DK_VEC; k++) {
)"
		R"(            dot_acc = mad(q_priv[k], CONVERT_ACC4(k_ptr[k]), dot_acc);
)"
		R"(        }
)"
		R"(        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
)"
		R"(        if (mask_base != NULL) {
)"
		R"(            const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base);
)"
		R"(            score += slope * (ACC_TYPE)mask_ptr[k_idx];
)"
		R"(        }
)"
		R"(        if (logit_softcap > 0.0f) {
)"
		R"(            score = logit_softcap * tanh(score / logit_softcap);
)"
		R"(        }
)"
		R"(        m_i = max(m_i, score);
)"
		R"(    }
)"
		R"(
)"
		R"(    __local ACC_TYPE local_m[Q1_WG_SIZE];
)"
		R"(    local_m[tid] = m_i;
)"
		R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
		R"(    #pragma unroll
)"
		R"(    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
)"
		R"(        if (tid < s) local_m[tid] = max(local_m[tid], local_m[tid + s]);
)"
		R"(        barrier(CLK_LOCAL_MEM_FENCE);
)"
		R"(    }
)"
		R"(    const ACC_TYPE m_final = local_m[0];
)"
		R"(
)"
		R"(    ACC_TYPE4 o_acc[DV_VEC];
)"
		R"(    #pragma unroll
)"
		R"(    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);
)"
		R"(    ACC_TYPE l_i = 0.0f;
)"
		R"(
)"
		R"(    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
)"
		R"(        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
)"
		R"(        const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
)"
		R"(        const global DATA_TYPE4* k_ptr = (const global DATA_TYPE4*)(k_base + k_row_offset);
)"
		R"(        const global DATA_TYPE4* v_ptr = (const global DATA_TYPE4*)(v_base + v_row_offset);
)"
		R"(        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
)"
		R"(        #pragma unroll
)"
		R"(        for (int k = 0; k < DK_VEC; k++) {
)"
		R"(            dot_acc = mad(q_priv[k], CONVERT_ACC4(k_ptr[k]), dot_acc);
)"
		R"(        }
)"
		R"(        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
)"
		R"(        if (mask_base != NULL) {
)"
		R"(            const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base);
)"
		R"(            score += slope * (ACC_TYPE)mask_ptr[k_idx];
)"
		R"(        }
)"
		R"(        if (logit_softcap > 0.0f) {
)"
		R"(            score = logit_softcap * tanh(score / logit_softcap);
)"
		R"(        }
)"
		R"(        const ACC_TYPE p = exp(score - m_final);
)"
		R"(        l_i += p;
)"
		R"(        #pragma unroll
)"
		R"(        for (int i = 0; i < DV_VEC; i++) {
)"
		R"(            o_acc[i] = mad(p, CONVERT_ACC4(v_ptr[i]), o_acc[i]);
)"
		R"(        }
)"
		R"(    }
)"
		R"(
)"
		R"(    __local ACC_TYPE local_l[Q1_WG_SIZE];
)"
		R"(    __local ACC_TYPE4 local_o_comp[Q1_WG_SIZE];
)"
		R"(    local_l[tid] = l_i;
)"
		R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
		R"(    #pragma unroll
)"
		R"(    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
)"
		R"(        if (tid < s) local_l[tid] += local_l[tid + s];
)"
		R"(        barrier(CLK_LOCAL_MEM_FENCE);
)"
		R"(    }
)"
		R"(
)"
		R"(    const ulong o_row_offset = batch_idx * o_nb3 + head_idx * o_nb1;
)"
		R"(    global DATA_TYPE4 *o_row = (global DATA_TYPE4 *)(o_base + o_row_offset);
)"
		R"(    ACC_TYPE l_final = local_l[0];
)"
		R"(
)"
		R"(    if (sinks_ptr != NULL) {
)"
		R"(        l_final += exp(sinks_ptr[head_idx] - m_final);
)"
		R"(    }
)"
		R"(
)"
		R"(    if (l_final > 0.0f) {
)"
		R"(        const ACC_TYPE l_inv = 1.0f / l_final;
)"
		R"(        for (int i = 0; i < DV_VEC; i++) {
)"
		R"(            local_o_comp[tid] = o_acc[i];
)"
		R"(            barrier(CLK_LOCAL_MEM_FENCE);
)"
		R"(            #pragma unroll
)"
		R"(            for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
)"
		R"(                if (tid < s) local_o_comp[tid] += local_o_comp[tid + s];
)"
		R"(                barrier(CLK_LOCAL_MEM_FENCE);
)"
		R"(            }
)"
		R"(            if (tid == 0) {
)"
		R"(                o_row[i] = CONVERT_DATA4(local_o_comp[0] * l_inv);
)"
		R"(            }
)"
		R"(        }
)"
		R"(    } else if (tid == 0) {
)"
		R"(        #pragma unroll
)"
		R"(        for (int i = 0; i < DV_VEC; ++i) o_row[i] = (DATA_TYPE4)(0.0f);
)"
		R"(    }
)"
		R"(}
)"
