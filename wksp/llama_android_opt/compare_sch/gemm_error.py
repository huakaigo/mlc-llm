T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
n = T.int32()
lv45 = T.match_buffer(p_lv45, (1, n, 4096), "float16")
p_output0_intermediate = T.match_buffer(p_output0, (1, n, w_y), "float16")
# with T.block("root"):
decode_local = T.alloc_buffer((4096, w_y), "float16", scope="local")
lv36_local = T.alloc_buffer((512, w_y), "uint32", scope="local")
lv37_local = T.alloc_buffer((128, w_y), "float16", scope="local")
lv45_pad_local = T.alloc_buffer(
    (1, (n + 31) // 32 * 32, 4096), "float16", scope="local"
)
var_NT_matmul_intermediate_pad_local = T.alloc_buffer(
    (1, (n + 31) // 32 * 32, w_y), "float16", scope="local"
)

# 任务划分:
### 一个thread处理 `processed_rows_per_thread`行 `vectorize_factor` 列(输出角度)
### 完整处理 `processed_rows_per_thread` 行输入需要: blockIdx.x * threadIdx.x 配合
### 完整处理 `n` 行输入需要: blockIdx.y * threadIdx.y 配合
#### 分析: 根据`n`变化的只有 blockIdx.y, 说明 blockIdx.x * threadIdx.x * threadIdx.y 可以完整处理32行输入
#  4 16 24 128 2
BlockIdx_x = 86
# n = 32
# BlockIdx_y = (n+31)//32 * 32 # 这里32是假设输入为32的倍数, //32的32 = thready * 
ThreadIdx_x = 16
ThreadIdx_y = 8
vectorize_factor = 8
processed_columns_per_thread = vectorize_factor# w_y / (BlockIdx_x * ThreadIdx_x) == vectorize_factor
processed_rows_per_thread = 4# == 32 / threadIdx.y

for i0_i1_fused_0_i0_i1_fused_1_0_fused in T.thread_binding(
    (n + 31) // 32, thread="blockIdx.y"
):
    for i2_0 in T.thread_binding(BlockIdx_x, thread="blockIdx.x"):
        for i0_i1_fused_1_1 in T.thread_binding(ThreadIdx_y, thread="threadIdx.y"):
            for i2_1 in T.thread_binding(ThreadIdx_x, thread="threadIdx.x"):
                for i0_i1_fused_1_2_init in range(processed_rows_per_thread):
                    for i2_2_init in T.vectorized(vectorize_factor):
                        with T.block("NT_matmul_init"):
                            v_i0 = T.axis.spatial(1, 0)
                            v_i1 = T.axis.spatial(
                                (n + 31) // 32 * 32,
                                i0_i1_fused_0_i0_i1_fused_1_0_fused * 32
                                + i0_i1_fused_1_1 * processed_rows_per_thread
                                + i0_i1_fused_1_2_init,
                            )
                            v_i2 = T.axis.spatial(
                                w_y, i2_0 * (ThreadIdx_x * vectorize_factor) + i2_1 * vectorize_factor + i2_2_init
                            )
                            T.reads()
                            T.writes(
                                var_NT_matmul_intermediate_pad_local[
                                    v_i0, v_i1, v_i2
                                ]
                            )
                            var_NT_matmul_intermediate_pad_local[
                                v_i0, v_i1, v_i2
                            ] = T.float16(0)
                for k_0 in range(128):
                    for ax0 in range(1):
                        for ax1 in T.vectorized(vectorize_factor):
                            with T.block("lv37_local"):
                                v0 = T.axis.spatial(128, k_0 + ax0)
                                v1 = T.axis.spatial(
                                    w_y, i2_0 * (ThreadIdx_x * vectorize_factor) + i2_1 * vectorize_factor + ax1
                                )
                                T.reads(lv37[v0, v1])
                                T.writes(lv37_local[v0, v1])
                                lv37_local[v0, v1] = lv37[v0, v1]
                    for k_1 in range(4):
                        for ax0 in range(1):
                            for ax1 in T.vectorized(vectorize_factor):
                                with T.block("lv36_local"):
                                    v0 = T.axis.spatial(512, k_0 * 4 + k_1 + ax0)
                                    v1 = T.axis.spatial(
                                        w_y, i2_0 * (ThreadIdx_x * vectorize_factor) + i2_1 * vectorize_factor + ax1
                                    )
                                    T.reads(lv36[v0, v1])
                                    T.writes(lv36_local[v0, v1])
                                    lv36_local[v0, v1] = lv36[v0, v1]
                        for k_2 in range(8):
                            for ax0 in range(1):
                                for ax1 in T.vectorized(vectorize_factor):
                                    with T.block("decode"):
                                        v_i = T.axis.spatial(
                                            4096, k_0 * 32 + k_1 * 8 + k_2 + ax0
                                        )
                                        v_j = T.axis.spatial(
                                            w_y, i2_0 * (ThreadIdx_x * vectorize_factor) + i2_1 * vectorize_factor + ax1
                                        )
                                        T.reads(
                                            lv36_local[v_i // 8, v_j],
                                            lv37_local[v_i // 32, v_j],
                                        )
                                        T.writes(decode_local[v_i, v_j])
                                        decode_local[v_i, v_j] = (
                                            T.Cast(
                                                "float16",
                                                T.bitwise_and(
                                                    T.shift_right(
                                                        lv36_local[v_i // 8, v_j],
                                                        T.Cast("uint32", v_i % 8)
                                                        * T.uint32(4),
                                                    ),
                                                    T.uint32(15),
                                                ),
                                            )
                                            - T.float16(7)
                                        ) * lv37_local[v_i // 32, v_j]
                            for ax0, ax1 in T.grid(1, processed_rows_per_thread):
                                for ax2 in T.vectorized(1):
                                    with T.block("lv45_pad_local"):
                                        v0 = T.axis.spatial(1, ax0)
                                        v1 = T.axis.spatial(
                                            (n + 31) // 32 * 32,
                                            i0_i1_fused_0_i0_i1_fused_1_0_fused * 32
                                            + i0_i1_fused_1_1 * processed_rows_per_thread
                                            + ax1,
                                        )
                                        v2 = T.axis.spatial(
                                            4096, k_0 * 32 + k_1 * 8 + k_2 + ax2
                                        )
                                        T.reads(lv45[v0, v1, v2])
                                        T.writes(lv45_pad_local[v0, v1, v2])
                                        lv45_pad_local[v0, v1, v2] = T.if_then_else(
                                            v1 < n, lv45[v0, v1, v2], T.float16(0)
                                        )
                            for i0_i1_fused_1_2 in range(processed_rows_per_thread):
                                for i2_2 in T.vectorized(vectorize_factor):
                                    with T.block("NT_matmul_update"):
                                        v_i0 = T.axis.spatial(1, 0)
                                        v_i1 = T.axis.spatial(
                                            (n + 31) // 32 * 32,
                                            i0_i1_fused_0_i0_i1_fused_1_0_fused * 32
                                            + i0_i1_fused_1_1 * processed_rows_per_thread
                                            + i0_i1_fused_1_2,
                                        )
                                        v_i2 = T.axis.spatial(
                                            w_y, i2_0 * (ThreadIdx_x * vectorize_factor) + i2_1 * vectorize_factor + i2_2
                                        )
                                        v_k = T.axis.reduce(
                                            4096, k_0 * 32 + k_1 * 8 + k_2
                                        )
                                        T.reads(
                                            var_NT_matmul_intermediate_pad_local[
                                                v_i0, v_i1, v_i2
                                            ],
                                            lv45_pad_local[v_i0, v_i1, v_k],
                                            decode_local[v_k, v_i2],
                                        )
                                        T.writes(
                                            var_NT_matmul_intermediate_pad_local[
                                                v_i0, v_i1, v_i2
                                            ]
                                        )
                                        var_NT_matmul_intermediate_pad_local[
                                            v_i0, v_i1, v_i2
                                        ] = (
                                            var_NT_matmul_intermediate_pad_local[
                                                v_i0, v_i1, v_i2
                                            ]
                                            + lv45_pad_local[v_i0, v_i1, v_k]
                                            * decode_local[v_k, v_i2]
                                        )
                for ax0, ax1 in T.grid(1, processed_rows_per_thread):
                    for ax2 in T.vectorized(vectorize_factor):
                        with T.block("var_NT_matmul_intermediate_pad_local"):
                            v0 = T.axis.spatial(1, ax0)
                            v1 = T.axis.spatial(
                                (n + 31) // 32 * 32,
                                i0_i1_fused_0_i0_i1_fused_1_0_fused * 32
                                + i0_i1_fused_1_1 * processed_rows_per_thread
                                + ax1,
                            )
                            v2 = T.axis.spatial(w_y, i2_0 * (ThreadIdx_x * vectorize_factor) + i2_1 * vectorize_factor + ax2)
                            T.reads(
                                var_NT_matmul_intermediate_pad_local[v0, v1, v2]
                            )
                            T.writes(p_output0_intermediate[v0, v1, v2])
                            if v1 < n:
                                p_output0_intermediate[
                                    v0, v1, v2
                                ] = var_NT_matmul_intermediate_pad_local[
                                    v0, v1, v2
                                ]