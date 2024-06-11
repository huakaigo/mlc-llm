# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(lv1323: T.Buffer((T.int64(512), T.int64(49984)), "uint32"), lv1324: T.Buffer((T.int64(128), T.int64(49984)), "float16"), lv1607: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(49984)), "float32")):
        T.func_attr({"global_symbol": "main", "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(49984)), "float16", scope="local")
        lv1323_local = T.alloc_buffer((T.int64(512), T.int64(49984)), "uint32", scope="local")
        lv1324_local = T.alloc_buffer((T.int64(128), T.int64(49984)), "float16", scope="local")
        lv1607_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16", scope="shared")
        for i0_i1_i2_fused_0 in T.thread_binding(T.int64(16), thread="blockIdx.x"):
            for i0_i1_i2_fused_1 in T.thread_binding(T.int64(781), thread="threadIdx.x"):
                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(21)):
                    for ax2_1 in T.thread_binding(T.int64(100), thread="threadIdx.x"):
                        for ax2_2 in T.vectorized(T.int64(2)):
                            with T.block("lv1607_shared"):
                                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                                v2 = T.axis.spatial(T.int64(4096), ax2_0 * T.int64(200) + ax2_1 * T.int64(2) + ax2_2)
                                T.where((ax2_0 * T.int64(100) + ax2_1) * T.int64(2) + ax2_2 < T.int64(4096))
                                T.reads(lv1607[v0, v1, v2])
                                T.writes(lv1607_shared[v0, v1, v2])
                                lv1607_shared[v0, v1, v2] = lv1607[v0, v1, v2]
                for i0_i1_i2_fused_2_init in T.vectorized(T.int64(4)):
                    with T.block("matmul_init"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i2 = T.axis.spatial(T.int64(49984), i0_i1_i2_fused_0 * T.int64(3124) + i0_i1_i2_fused_1 * T.int64(4) + i0_i1_i2_fused_2_init)
                        T.reads()
                        T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                        var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float16(0)
                for k_0 in range(T.int64(512)):
                    for ax0 in range(T.int64(1)):
                        for ax1 in T.vectorized(T.int64(4)):
                            with T.block("lv1324_local"):
                                v0 = T.axis.spatial(T.int64(128), k_0 // T.int64(4) + ax0)
                                v1 = T.axis.spatial(T.int64(49984), i0_i1_i2_fused_0 * T.int64(3124) + i0_i1_i2_fused_1 * T.int64(4) + ax1)
                                T.reads(lv1324[v0, v1])
                                T.writes(lv1324_local[v0, v1])
                                lv1324_local[v0, v1] = lv1324[v0, v1]
                    for k_1 in range(T.int64(8)):
                        for ax0 in range(T.int64(1)):
                            for ax1 in T.vectorized(T.int64(4)):
                                with T.block("lv1323_local"):
                                    v0 = T.axis.spatial(T.int64(512), k_0 + ax0)
                                    v1 = T.axis.spatial(T.int64(49984), i0_i1_i2_fused_0 * T.int64(3124) + i0_i1_i2_fused_1 * T.int64(4) + ax1)
                                    T.reads(lv1323[v0, v1])
                                    T.writes(lv1323_local[v0, v1])
                                    lv1323_local[v0, v1] = lv1323[v0, v1]
                        for k_2 in range(T.int64(1)):
                            for i0_i1_i2_fused_2 in T.vectorized(T.int64(4)):
                                with T.block("matmul_update"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i2 = T.axis.spatial(T.int64(49984), i0_i1_i2_fused_0 * T.int64(3124) + i0_i1_i2_fused_1 * T.int64(4) + i0_i1_i2_fused_2)
                                    v_k = T.axis.reduce(T.int64(4096), k_0 * T.int64(8) + k_1 + k_2)
                                    T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv1607_shared[v_i0, v_i1, v_k], lv1323_local[v_k // T.int64(8), v_i2], lv1324_local[v_k // T.int64(32), v_i2])
                                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv1607_shared[v_i0, v_i1, v_k] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv1323_local[v_k // T.int64(8), v_i2], T.Cast("uint32", v_k % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv1324_local[v_k // T.int64(32), v_i2])
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("var_matmul_intermediate_local"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(T.int64(49984), i0_i1_i2_fused_0 * T.int64(3124) + i0_i1_i2_fused_1 * T.int64(4) + ax2)
                            T.reads(var_matmul_intermediate_local[v0, v1, v2])
                            T.writes(p_output0_intermediate[v0, v1, v2])
                            p_output0_intermediate[v0, v1, v2] = T.Cast("float32", var_matmul_intermediate_local[v0, v1, v2])