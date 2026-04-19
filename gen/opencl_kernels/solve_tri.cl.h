R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
		R"(
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(// solve_tri
)"
		R"(//------------------------------------------------------------------------------
)"
		R"(kernel void kernel_solve_tri_f32(
)"
		R"(        global uchar * src0,
)"
		R"(        ulong offset0,
)"
		R"(        global uchar * src1,
)"
		R"(        ulong offset1,
)"
		R"(        global uchar * dst,
)"
		R"(        ulong offsetd,
)"
		R"(        int n,
)"
		R"(        int k,
)"
		R"(        ulong nb00,
)"
		R"(        ulong nb01,
)"
		R"(        ulong nb02,
)"
		R"(        ulong nb03,
)"
		R"(        ulong nb10,
)"
		R"(        ulong nb11,
)"
		R"(        ulong nb12,
)"
		R"(        ulong nb13,
)"
		R"(        ulong nb0,
)"
		R"(        ulong nb1,
)"
		R"(        ulong nb2,
)"
		R"(        ulong nb3
)"
		R"() {
)"
		R"(    int col = get_global_id(0);
)"
		R"(    int i2 = get_global_id(1);
)"
		R"(    int i3 = get_global_id(2);
)"
		R"(
)"
		R"(    global const uchar * Lb = src0 + offset0 + i2 * nb02 + i3 * nb03;
)"
		R"(    global const uchar * Bb = src1 + offset1 + i2 * nb12 + i3 * nb13;
)"
		R"(    global       uchar * Xb = dst + offsetd + i2 * nb2 + i3 * nb3;
)"
		R"(
)"
		R"(    for(int row = 0; row < n; ++row){
)"
		R"(        global const float *pB = (global const float *)(Bb + row * nb11 + col * nb10);
)"
		R"(
)"
		R"(        float sum = 0.0f;
)"
		R"(        for(int j = 0; j < row; ++j){
)"
		R"(            global const float *pL = (global const float *)(Lb + row * nb01 + j * nb00);
)"
		R"(            global const float *pX = (global const float *)(Xb + j * nb1 + col * nb0);
)"
		R"(            sum += (*pL) * (*pX);
)"
		R"(        }
)"
		R"(
)"
		R"(        global const float * pDiag = (global const float *)(Lb + row * nb01 + row *nb00);
)"
		R"(        global float * pOut = (global float *)(Xb + row * nb1 + col *nb0);
)"
		R"(
)"
		R"(        *pOut = ((* pB) - sum) / (*pDiag);
)"
		R"(    }
)"
		R"(}
)"
