R"(kernel void kernel_timestep_embedding(
)"
		R"(    global const void * p_timesteps,
)"
		R"(    ulong off_timesteps,
)"
		R"(    global void * p_dst,
)"
		R"(    ulong off_dst,
)"
		R"(    int dst_nb1_bytes,
)"
		R"(    int logical_dim,
)"
		R"(    int max_period
)"
		R"() {
)"
		R"(    int local_i;
)"
		R"(    int local_j;
)"
		R"(    int local_half_dim;
)"
		R"(    float local_timestep_val;
)"
		R"(    float local_freq;
)"
		R"(    float local_arg;
)"
		R"(    global float * local_embed_data_ptr;
)"
		R"(    global const float * local_timesteps_input_ptr;
)"
		R"(    global float * local_dst_output_base_ptr;
)"
		R"(
)"
		R"(    local_timesteps_input_ptr = (global const float *)((global char *)p_timesteps + off_timesteps);
)"
		R"(    local_dst_output_base_ptr = (global float *)((global char *)p_dst + off_dst);
)"
		R"(
)"
		R"(    local_i = get_global_id(1);
)"
		R"(    local_j = get_global_id(0);
)"
		R"(
)"
		R"(    local_half_dim = logical_dim / 2;
)"
		R"(    local_embed_data_ptr = (global float *)((global char *)local_dst_output_base_ptr + local_i * dst_nb1_bytes);
)"
		R"(
)"
		R"(    if (logical_dim % 2 != 0 && local_j == local_half_dim) {
)"
		R"(        local_embed_data_ptr[2 * local_half_dim] = 0.0f;
)"
		R"(    }
)"
		R"(
)"
		R"(    if (local_j >= local_half_dim) {
)"
		R"(        return;
)"
		R"(    }
)"
		R"(
)"
		R"(    local_timestep_val = local_timesteps_input_ptr[local_i];
)"
		R"(
)"
		R"(    if (local_half_dim == 0) {
)"
		R"(        local_freq = 1.0f;
)"
		R"(    } else {
)"
		R"(        local_freq = exp(-log((float)max_period) * (float)local_j / (float)local_half_dim);
)"
		R"(    }
)"
		R"(
)"
		R"(    local_arg = local_timestep_val * local_freq;
)"
		R"(    local_embed_data_ptr[local_j] = cos(local_arg);
)"
		R"(    local_embed_data_ptr[local_j + local_half_dim] = sin(local_arg);
)"
		R"(}
)"
