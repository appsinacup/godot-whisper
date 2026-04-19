R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
		R"(
)"
		R"(// 16-bit transpose, loading/storing a 4x4 tile of elements
)"
		R"(kernel void kernel_transpose_16(
)"
		R"(    __read_only image1d_buffer_t input,
)"
		R"(    __write_only image1d_buffer_t output,
)"
		R"(    const uint rows,
)"
		R"(    const uint cols
)"
		R"() {
)"
		R"(
)"
		R"(    const int i = get_global_id(0);
)"
		R"(    const int j = get_global_id(1);
)"
		R"(    const int i_2 = i<<2;
)"
		R"(    const int j_2 = j<<2;
)"
		R"(
)"
		R"(    half4 temp0 = read_imageh(input, (j_2+0)*cols+i);
)"
		R"(    half4 temp1 = read_imageh(input, (j_2+1)*cols+i);
)"
		R"(    half4 temp2 = read_imageh(input, (j_2+2)*cols+i);
)"
		R"(    half4 temp3 = read_imageh(input, (j_2+3)*cols+i);
)"
		R"(
)"
		R"(    write_imageh(output, (i_2+0)*rows+j, (half4)(temp0.s0, temp1.s0, temp2.s0, temp3.s0));
)"
		R"(    write_imageh(output, (i_2+1)*rows+j, (half4)(temp0.s1, temp1.s1, temp2.s1, temp3.s1));
)"
		R"(    write_imageh(output, (i_2+2)*rows+j, (half4)(temp0.s2, temp1.s2, temp2.s2, temp3.s2));
)"
		R"(    write_imageh(output, (i_2+3)*rows+j, (half4)(temp0.s3, temp1.s3, temp2.s3, temp3.s3));
)"
		R"(}
)"
		R"(
)"
		R"(// Padded kernel for irregular shape
)"
		R"(kernel void kernel_transpose_16_4x1(
)"
		R"(    __read_only image1d_buffer_t input,
)"
		R"(    __write_only image1d_buffer_t output,
)"
		R"(    const uint rows,
)"
		R"(    const uint cols
)"
		R"() {
)"
		R"(
)"
		R"(    const int i = get_global_id(0);
)"
		R"(    const int j = get_global_id(1);
)"
		R"(    const int j_2 = j << 2;
)"
		R"(
)"
		R"(    half temp0 = read_imageh(input, (j_2 + 0) * cols + i).x;
)"
		R"(    half temp1 = read_imageh(input, (j_2 + 1) * cols + i).x;
)"
		R"(    half temp2 = read_imageh(input, (j_2 + 2) * cols + i).x;
)"
		R"(    half temp3 = read_imageh(input, (j_2 + 3) * cols + i).x;
)"
		R"(
)"
		R"(    write_imageh(output, i * rows + j, (half4)(temp0, temp1, temp2, temp3));
)"
		R"(}
)"
		R"(
)"
		R"(// Transpose treating each element as 8-bit using buffer
)"
		R"(kernel void kernel_transpose_8_buf(
)"
		R"(    global const uchar * input,
)"
		R"(    global uchar * output,
)"
		R"(    const int ldi,
)"
		R"(    const int ldo
)"
		R"() {
)"
		R"(    const int x = get_global_id(0);
)"
		R"(    const int y = get_global_id(1);
)"
		R"(
)"
		R"(    output[x*ldo + y] = input[y*ldi + x];
)"
		R"(}
)"
		R"(
)"
		R"(// Transpose treating each element as 16-bit using buffer
)"
		R"(kernel void kernel_transpose_16_buf(
)"
		R"(    global const ushort * input,
)"
		R"(    global ushort * output,
)"
		R"(    const int ldi,
)"
		R"(    const int ldo
)"
		R"() {
)"
		R"(    const int x = get_global_id(0);
)"
		R"(    const int y = get_global_id(1);
)"
		R"(
)"
		R"(    output[x*ldo + y] = input[y*ldi + x];
)"
		R"(}
)"
		R"(
)"
		R"(// Transpose treating each element as 32-bit using buffer
)"
		R"(kernel void kernel_transpose_32_buf(
)"
		R"(    global const uint * input,
)"
		R"(    global uint * output,
)"
		R"(    const int ldi,
)"
		R"(    const int ldo
)"
		R"() {
)"
		R"(    const int x = get_global_id(0);
)"
		R"(    const int y = get_global_id(1);
)"
		R"(
)"
		R"(    output[x*ldo + y] = input[y*ldi + x];
)"
		R"(}
)"
		R"(
)"
		R"(// 32-bit transpose, loading/storing a 4x4 tile of elements
)"
		R"(kernel void kernel_transpose_32(
)"
		R"(    __read_only image1d_buffer_t input,
)"
		R"(    __write_only image1d_buffer_t output,
)"
		R"(    const uint rows,
)"
		R"(    const uint cols
)"
		R"() {
)"
		R"(
)"
		R"(    const int i = get_global_id(0);
)"
		R"(    const int j = get_global_id(1);
)"
		R"(    const int i_2 = i<<2;
)"
		R"(    const int j_2 = j<<2;
)"
		R"(
)"
		R"(    float4 temp0 = read_imagef(input, (j_2+0)*cols+i);
)"
		R"(    float4 temp1 = read_imagef(input, (j_2+1)*cols+i);
)"
		R"(    float4 temp2 = read_imagef(input, (j_2+2)*cols+i);
)"
		R"(    float4 temp3 = read_imagef(input, (j_2+3)*cols+i);
)"
		R"(
)"
		R"(    write_imagef(output, (i_2+0)*rows+j, (float4)(temp0.s0, temp1.s0, temp2.s0, temp3.s0));
)"
		R"(    write_imagef(output, (i_2+1)*rows+j, (float4)(temp0.s1, temp1.s1, temp2.s1, temp3.s1));
)"
		R"(    write_imagef(output, (i_2+2)*rows+j, (float4)(temp0.s2, temp1.s2, temp2.s2, temp3.s2));
)"
		R"(    write_imagef(output, (i_2+3)*rows+j, (float4)(temp0.s3, temp1.s3, temp2.s3, temp3.s3));
)"
		R"(
)"
		R"(}
)"
		R"(
)"
		R"(// 32-bit transpose, loading/storing a 4x4 tile of elements
)"
		R"(// Only used for activations
)"
		R"(// converts to FP16
)"
		R"(// also adds zero padding for non multiple of 8 prompt lengths
)"
		R"(kernel void kernel_transpose_32_16(__read_only image1d_buffer_t input, __write_only image1d_buffer_t output, const uint rows, const uint cols, const uint padded_rows) {
)"
		R"(
)"
		R"(    const int i = get_global_id(0);
)"
		R"(    const int j = get_global_id(1);
)"
		R"(    const int i_2 = i<<2;
)"
		R"(    const int j_2 = j<<2;
)"
		R"(    half4 temp0 = {0,0,0,0}; // initialize outputs to 0
)"
		R"(    half4 temp1 = {0,0,0,0};
)"
		R"(    half4 temp2 = {0,0,0,0};
)"
		R"(    half4 temp3 = {0,0,0,0};
)"
		R"(
)"
		R"(    if((j_2+0)*cols+i*4+3 < rows*cols*16){ // only load from a valid location. Otherwise keep register data as 0
)"
		R"(        temp0 = read_imageh(input, (j_2+0)*cols+i);
)"
		R"(    }
)"
		R"(    if((j_2+1)*cols+i*4+3 < rows*cols*16){
)"
		R"(        temp1 = read_imageh(input, (j_2+1)*cols+i);
)"
		R"(    }
)"
		R"(    if((j_2+2)*cols+i*4+3 < rows*cols*16){
)"
		R"(        temp2 = read_imageh(input, (j_2+2)*cols+i);
)"
		R"(    }
)"
		R"(    if((j_2+3)*cols+i*4+3 < rows*cols*16){
)"
		R"(        temp3 = read_imageh(input, (j_2+3)*cols+i);
)"
		R"(    }
)"
		R"(
)"
		R"(    write_imageh(output, (i_2+0)*padded_rows+j, (half4)(temp0.s0, temp1.s0, temp2.s0, temp3.s0)); // no conditionals for output, includes zero padding
)"
		R"(    write_imageh(output, (i_2+1)*padded_rows+j, (half4)(temp0.s1, temp1.s1, temp2.s1, temp3.s1));
)"
		R"(    write_imageh(output, (i_2+2)*padded_rows+j, (half4)(temp0.s2, temp1.s2, temp2.s2, temp3.s2));
)"
		R"(    write_imageh(output, (i_2+3)*padded_rows+j, (half4)(temp0.s3, temp1.s3, temp2.s3, temp3.s3));
)"
		R"(}
)"
