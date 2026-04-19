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
R"(#define OPWM 64
)"
R"(#define OPWN 64
)"
R"(#define CPWK 8
)"
R"(#define OPTM 4
)"
R"(#define OPTN 8
)"
R"(
)"
R"(#define WG_M (OPWM / OPTM)
)"
R"(#define WG_N (OPWN / OPTN)
)"
R"(#define VEC_K (CPWK / 4)
)"
R"(
)"
R"(REQD_SUBGROUP_SIZE_128
)"
R"(__kernel void mul_mat_f16_f32(
)"
R"(    const int M, const int N, const int K,
)"
R"(    __global const void* A_void, ulong A_offset,
)"
R"(    __global const void* B_void, ulong B_offset,
)"
R"(    __global       void* C_void, ulong C_offset) {
)"
R"(
)"
R"(    __global const half*  A = (__global const half* )((__global const char*)A_void + A_offset);
)"
R"(    __global const float* B = (__global const float*)((__global const char*)B_void + B_offset);
)"
R"(    __global       float* C = (__global       float*)((__global       char*)C_void + C_offset);
)"
R"(
)"
R"(    const int lidm = get_local_id(0);
)"
R"(    const int lidn = get_local_id(1);
)"
R"(    const int lid = lidn * WG_M + lidm;
)"
R"(
)"
R"(    const int offsetM = get_group_id(0) * OPWM;
)"
R"(    const int offsetN = get_group_id(1) * OPWN;
)"
R"(
)"
R"(    __local half4  Alocal[OPWM][VEC_K];
)"
R"(    __local float4 Blocal[OPWN][VEC_K];
)"
R"(
)"
R"(    float sum[OPTM][OPTN];
)"
R"(
)"
R"(    for (int wm = 0; wm < OPTM; wm++) {
)"
R"(        for (int wn = 0; wn < OPTN; wn++) {
)"
R"(            sum[wm][wn] = 0.0f;
)"
R"(        }
)"
R"(    }
)"
R"(
)"
R"(    const int numTiles = (K + CPWK - 1) / CPWK;
)"
R"(
)"
R"(    const int load_row_a = lid % OPWM;
)"
R"(    const int load_vec_k_a = lid / OPWM;
)"
R"(    const int global_row_a = offsetM + load_row_a;
)"
R"(
)"
R"(    const int load_row_b = lid % OPWN;
)"
R"(    const int load_vec_k_b = lid / OPWN;
)"
R"(    const int global_row_b = offsetN + load_row_b;
)"
R"(
)"
R"(    for (int t = 0; t < numTiles; t++) {
)"
R"(        const int k_start = t * CPWK;
)"
R"(        const int k_vec_start_a = k_start + load_vec_k_a * 4;
)"
R"(        const int k_vec_start_b = k_start + load_vec_k_b * 4;
)"
R"(
)"
R"(        if (global_row_a < M && k_vec_start_a < K) {
)"
R"(            if (k_vec_start_a + 3 < K) {
)"
R"(                Alocal[load_row_a][load_vec_k_a] = vload4(0, A + global_row_a * K + k_vec_start_a);
)"
R"(            } else {
)"
R"(                half4 tempA = (half4)(0.0h);
)"
R"(                if (k_vec_start_a < K) tempA.s0 = A[global_row_a * K + k_vec_start_a];
)"
R"(                if (k_vec_start_a + 1 < K) tempA.s1 = A[global_row_a * K + k_vec_start_a + 1];
)"
R"(                if (k_vec_start_a + 2 < K) tempA.s2 = A[global_row_a * K + k_vec_start_a + 2];
)"
R"(                Alocal[load_row_a][load_vec_k_a] = tempA;
)"
R"(            }
)"
R"(        } else {
)"
R"(            Alocal[load_row_a][load_vec_k_a] = (half4)(0.0h);
)"
R"(        }
)"
R"(
)"
R"(        if (global_row_b < N && k_vec_start_b < K) {
)"
R"(            if (k_vec_start_b + 3 < K) {
)"
R"(                Blocal[load_row_b][load_vec_k_b] = vload4(0, B + global_row_b * K + k_vec_start_b);
)"
R"(            } else {
)"
R"(                float4 tempB = (float4)(0.0f);
)"
R"(                if (k_vec_start_b < K) tempB.s0 = B[global_row_b * K + k_vec_start_b];
)"
R"(                if (k_vec_start_b + 1 < K) tempB.s1 = B[global_row_b * K + k_vec_start_b + 1];
)"
R"(                if (k_vec_start_b + 2 < K) tempB.s2 = B[global_row_b * K + k_vec_start_b + 2];
)"
R"(                Blocal[load_row_b][load_vec_k_b] = tempB;
)"
R"(            }
)"
R"(        } else {
)"
R"(            Blocal[load_row_b][load_vec_k_b] = (float4)(0.0f);
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
R"(        for (int k_vec = 0; k_vec < VEC_K; k_vec++) {
)"
R"(            float4 a_fvecs[OPTM];
)"
R"(            int current_row_a = lidm;
)"
R"(            for (int wm = 0; wm < OPTM; wm++) {
)"
R"(                a_fvecs[wm] = convert_float4(Alocal[current_row_a][k_vec]);
)"
R"(                current_row_a += WG_M;
)"
R"(            }
)"
R"(
)"
R"(            float4 b_fvecs[OPTN];
)"
R"(            int current_row_b = lidn;
)"
R"(            for (int wn = 0; wn < OPTN; wn++) {
)"
R"(                b_fvecs[wn] = Blocal[current_row_b][k_vec];
)"
R"(                current_row_b += WG_N;
)"
R"(            }
)"
R"(
)"
R"(            for (int wm = 0; wm < OPTM; wm++) {
)"
R"(                for (int wn = 0; wn < OPTN; wn++) {
)"
R"(                    sum[wm][wn] += dot(a_fvecs[wm], b_fvecs[wn]);
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
R"(    for (int wm = 0; wm < OPTM; wm++) {
)"
R"(        int globalRow = offsetM + lidm + wm * WG_M;
)"
R"(        if (globalRow < M) {
)"
R"(            for (int wn = 0; wn < OPTN; wn++) {
)"
R"(                int globalCol = offsetN + lidn + wn * WG_N;
)"
R"(                if (globalCol < N) {
)"
R"(                    C[globalCol * M + globalRow] = sum[wm][wn];
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
