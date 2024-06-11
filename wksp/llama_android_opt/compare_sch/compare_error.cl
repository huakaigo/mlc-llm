// Function: fused_decode1_fused_NT_matmul2_silu_after_kernel
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

__kernel void fused_decode1_fused_NT_matmul2_silu_after_kernel(__global uint* restrict lv36, __global half* restrict lv37, __global half* restrict lv45, __global half* restrict p_output0_intermediate, int n) {
  half8 var_NT_matmul_intermediate_pad_local[4];
  half8 lv37_local[1];
  uint8 lv36_local[1];
  half8 decode_local[1];
  half lv45_pad_local[4];
  for (int i0_i1_fused_1_2_init = 0; i0_i1_fused_1_2_init < 4; ++i0_i1_fused_1_2_init) {
    var_NT_matmul_intermediate_pad_local[i0_i1_fused_1_2_init] = ((half8)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f));
  }
  for (int k_0 = 0; k_0 < 128; ++k_0) {
    lv37_local[0] = vload8(0, lv37 + (((k_0 * 22016) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)));
    for (int k_1 = 0; k_1 < 4; ++k_1) {
      lv36_local[0] = vload8(0, lv36 + ((((k_0 * 88064) + (k_1 * 22016)) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)));
      for (int k_2 = 0; k_2 < 8; ++k_2) {
        decode_local[0] = (((convert_half8(((lv36_local[0]  >>  ((uint8)(((convert_uint(k_2)) * (uint)4), ((convert_uint(k_2)) * (uint)4), ((convert_uint(k_2)) * (uint)4), ((convert_uint(k_2)) * (uint)4), ((convert_uint(k_2)) * (uint)4), ((convert_uint(k_2)) * (uint)4), ((convert_uint(k_2)) * (uint)4), ((convert_uint(k_2)) * (uint)4))))  &  ((uint8)((uint)15, (uint)15, (uint)15, (uint)15, (uint)15, (uint)15, (uint)15, (uint)15))))) - ((half8)((half)7.000000e+00f, (half)7.000000e+00f, (half)7.000000e+00f, (half)7.000000e+00f, (half)7.000000e+00f, (half)7.000000e+00f, (half)7.000000e+00f, (half)7.000000e+00f))) * lv37_local[0]);
        for (int ax1 = 0; ax1 < 4; ++ax1) {
          lv45_pad_local[ax1] = ((((((convert_int(get_group_id(1))) * 32) + ((convert_int(get_local_id(1))) * 4)) + ax1) < n) ? lv45[(((((((convert_int(get_group_id(1))) * 131072) + ((convert_int(get_local_id(1))) * 16384)) + (ax1 * 4096)) + (k_0 * 32)) + (k_1 * 8)) + k_2)] : (half)0.000000e+00f);
        }
        for (int i0_i1_fused_1_2 = 0; i0_i1_fused_1_2 < 4; ++i0_i1_fused_1_2) {
          var_NT_matmul_intermediate_pad_local[i0_i1_fused_1_2] = (var_NT_matmul_intermediate_pad_local[i0_i1_fused_1_2] + (((half8)(lv45_pad_local[i0_i1_fused_1_2], lv45_pad_local[i0_i1_fused_1_2], lv45_pad_local[i0_i1_fused_1_2], lv45_pad_local[i0_i1_fused_1_2], lv45_pad_local[i0_i1_fused_1_2], lv45_pad_local[i0_i1_fused_1_2], lv45_pad_local[i0_i1_fused_1_2], lv45_pad_local[i0_i1_fused_1_2])) * decode_local[0]));
        }
      }
    }
  }
  for (int ax1_1 = 0; ax1_1 < 4; ++ax1_1) {
    if (((((convert_int(get_group_id(1))) * 32) + ((convert_int(get_local_id(1))) * 4)) + ax1_1) < n) {
      vstore8(var_NT_matmul_intermediate_pad_local[ax1_1], 0, p_output0_intermediate + ((((((convert_int(get_group_id(1))) * 704512) + ((convert_int(get_local_id(1))) * 88064)) + (ax1_1 * 22016)) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)));
    }
  }
}