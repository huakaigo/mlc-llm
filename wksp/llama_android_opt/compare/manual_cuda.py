import tvm
from tvm.script import ir as I
from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(lv1323: T.Buffer((T.int64(512), T.int64(49984)), "uint32"), lv1324: T.Buffer((T.int64(128), T.int64(49984)), "float16"), lv1607: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(49984)), "float32")):
        T.func_attr({"global_symbol": "main", "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(49984)), "float16", scope="local")
        lv1607_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16", scope="shared")
        p_output0_intermediate_1_shared = T.alloc_buffer((T.int64(4096), T.int64(49984)), "float16", scope="shared")
        for i0_0_i1_0_i2_0_fused in T.thread_binding(T.int64(781), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 1024, "pragma_unroll_explicit": 1}):
            for i0_1_i1_1_i2_1_fused in T.thread_binding(T.int64(2), thread="vthread.x"):
                for i0_2_i1_2_i2_2_fused in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    for i0_3_init, i1_3_init, i2_3_init, i0_4_init, i1_4_init, i2_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("matmul_init"):
                            v_i0 = T.axis.spatial(T.int64(1), i0_3_init + i0_4_init)
                            v_i1 = T.axis.spatial(T.int64(1), i1_3_init + i1_4_init)
                            v_i2 = T.axis.spatial(T.int64(49984), i0_0_i1_0_i2_0_fused * T.int64(64) + i0_1_i1_1_i2_1_fused * T.int64(32) + i0_2_i1_2_i2_2_fused + i2_3_init + i2_4_init)
                            T.reads()
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                            var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float16(0)
                    for k_0 in range(T.int64(128)):
                        for ax0_ax1_ax2_fused_0 in range(T.int64(1)):
                            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_ax2_fused_2 in T.vectorized(T.int64(2)):
                                    with T.block("lv1607_shared"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2 = T.axis.spatial(T.int64(4096), k_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(2) + ax0_ax1_ax2_fused_2))
                                        T.where((ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1) * T.int64(2) + ax0_ax1_ax2_fused_2 < T.int64(32))
                                        T.reads(lv1607[v0, v1, v2])
                                        T.writes(lv1607_shared[v0, v1, v2])
                                        lv1607_shared[v0, v1, v2] = lv1607[v0, v1, v2]
                        for ax0_ax1_fused_0 in range(T.int64(32)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(T.int64(2)):
                                    with T.block("p_output0_intermediate_1_shared"):
                                        v0 = T.axis.spatial(T.int64(4096), k_0 * T.int64(32) + (ax0_ax1_fused_0 * T.int64(64) + ax0_ax1_fused_1 * T.int64(2) + ax0_ax1_fused_2) // T.int64(64))
                                        v1 = T.axis.spatial(T.int64(49984), i0_0_i1_0_i2_0_fused * T.int64(64) + (ax0_ax1_fused_0 * T.int64(64) + ax0_ax1_fused_1 * T.int64(2) + ax0_ax1_fused_2) % T.int64(64))
                                        T.reads(lv1323[v0 // T.int64(8), v1], lv1324[v0 // T.int64(32), v1])
                                        T.writes(p_output0_intermediate_1_shared[v0, v1])
                                        p_output0_intermediate_1_shared[v0, v1] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv1323[v0 // T.int64(8), v1], T.Cast("uint32", v0 % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv1324[v0 // T.int64(32), v1]
                        for k_1, i0_3, i1_3, i2_3, k_2, i0_4, i1_4, i2_4 in T.grid(T.int64(2), T.int64(1), T.int64(1), T.int64(1), T.int64(16), T.int64(1), T.int64(1), T.int64(1)):
                            with T.block("matmul_update"):
                                v_i0 = T.axis.spatial(T.int64(1), i0_3 + i0_4)
                                v_i1 = T.axis.spatial(T.int64(1), i1_3 + i1_4)
                                v_i2 = T.axis.spatial(T.int64(49984), i0_0_i1_0_i2_0_fused * T.int64(64) + i0_1_i1_1_i2_1_fused * T.int64(32) + i0_2_i1_2_i2_2_fused + i2_3 + i2_4)
                                v_k = T.axis.reduce(T.int64(4096), k_0 * T.int64(32) + k_1 * T.int64(16) + k_2)
                                T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv1607_shared[v_i0, v_i1, v_k], p_output0_intermediate_1_shared[v_k, v_i2])
                                T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv1607_shared[v_i0, v_i1, v_k] * p_output0_intermediate_1_shared[v_k, v_i2]
                    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("var_matmul_intermediate_local"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(T.int64(49984), i0_0_i1_0_i2_0_fused * T.int64(64) + i0_1_i1_1_i2_1_fused * T.int64(32) + i0_2_i1_2_i2_2_fused + ax2)
                            T.reads(var_matmul_intermediate_local[v0, v1, v2])
                            T.writes(p_output0_intermediate[v0, v1, v2])
                            p_output0_intermediate[v0, v1, v2] = T.Cast("float32", var_matmul_intermediate_local[v0, v1, v2])