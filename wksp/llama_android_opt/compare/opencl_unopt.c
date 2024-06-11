// Function: main_kernel_2
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

__kernel void main_kernel_2(__global float* restrict p_output0_intermediate, __global half* restrict var_matmul_intermediate) {
  if ((((convert_int(get_group_id(0))) * 4) + ((convert_int(get_local_id(0))) >> 6)) < 781) {
    p_output0_intermediate[(((convert_int(get_group_id(0))) * 256) + (convert_int(get_local_id(0))))] = (convert_float(var_matmul_intermediate[(((convert_int(get_group_id(0))) * 256) + (convert_int(get_local_id(0))))]));
  }
}

// Function: main_kernel_1
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

__kernel void main_kernel_1(__global half* restrict lv1607, __global half* restrict p_output0_intermediate_1, __global half* restrict var_matmul_intermediate) {
  for (int k = 0; k < 4096; ++k) {
    if ((((convert_int(get_group_id(0))) * 4) + ((convert_int(get_local_id(0))) >> 6)) < 781) {
      if (k == 0) {
        var_matmul_intermediate[(((convert_int(get_group_id(0))) * 256) + (convert_int(get_local_id(0))))] = (half)0.000000e+00f;
      }
      var_matmul_intermediate[(((convert_int(get_group_id(0))) * 256) + (convert_int(get_local_id(0))))] = (var_matmul_intermediate[(((convert_int(get_group_id(0))) * 256) + (convert_int(get_local_id(0))))] + (lv1607[k] * p_output0_intermediate_1[(((k * 49984) + ((convert_int(get_group_id(0))) * 256)) + (convert_int(get_local_id(0))))]));
    }
  }
}

// Function: main_kernel
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

__kernel void main_kernel(__global uint* restrict lv1323, __global half* restrict lv1324, __global half* restrict p_output0_intermediate_1) {
  for (int i_j_fused_0 = 0; i_j_fused_0 < 3124; ++i_j_fused_0) {
    p_output0_intermediate_1[(((i_j_fused_0 * 65536) + ((convert_int(get_group_id(0))) * 256)) + (convert_int(get_local_id(0))))] = (((convert_half(((lv1323[(((((i_j_fused_0 * 128) + ((convert_int(get_group_id(0))) >> 1)) / 781) * 49984) + ((((i_j_fused_0 * 65536) + ((convert_int(get_group_id(0))) * 256)) + (convert_int(get_local_id(0)))) % 49984))] >> ((convert_uint((((((i_j_fused_0 * 1024) + ((convert_int(get_group_id(0))) * 4)) + ((convert_int(get_local_id(0))) >> 6)) % 6248) / 781))) * (uint)4)) & (uint)15))) - (half)7.000000e+00f) * lv1324[(((((i_j_fused_0 * 32) + ((convert_int(get_group_id(0))) >> 3)) / 781) * 49984) + ((((i_j_fused_0 * 65536) + ((convert_int(get_group_id(0))) * 256)) + (convert_int(get_local_id(0)))) % 49984))]);
  }
}