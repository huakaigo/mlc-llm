import tvm
from tvm import ir as I
from tvm.script import tir as T

def get_func(vocab_size):
    @T.prim_func
    def fused_fused_decode11_fused_matmul9_cast2(lv1323: T.Buffer((T.int64(512), T.int64(vocab_size)), "uint32"), lv1324: T.Buffer((T.int64(128), T.int64(vocab_size)), "float16"), lv1607: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(vocab_size)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(vocab_size)), "float16")
        var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(vocab_size)), "float16")
        for i, j in T.grid(T.int64(4096), T.int64(vocab_size)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv1323[v_i // T.int64(8), v_j], lv1324[v_i // T.int64(32), v_j])
                T.writes(p_output0_intermediate_1[v_i, v_j])
                p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv1323[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv1324[v_i // T.int64(32), v_j]
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(vocab_size), T.int64(4096)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv1607[v_i0, v_i1, v_k], p_output0_intermediate_1[v_k, v_i2])
                T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1607[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_k, v_i2]
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(vocab_size)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
                T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
                p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", var_matmul_intermediate[v_i0, v_i1, v_i2])
    return fused_fused_decode11_fused_matmul9_cast2

@T.prim_func
def fused_fused_decode11_fused_matmul6_cast2(lv1323: T.Buffer((T.int64(512), T.int64(50000)), "uint32"), lv1324: T.Buffer((T.int64(128), T.int64(50000)), "float16"), lv1607: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(50000)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(50000)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(50000)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(50000)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1323[v_i // T.int64(8), v_j], lv1324[v_i // T.int64(32), v_j])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv1323[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv1324[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(50000), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1607[v_i0, v_i1, v_k], p_output0_intermediate_1[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1607[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_k, v_i2]
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(50000)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", var_matmul_intermediate[v_i0, v_i1, v_i2])


def get_dict_key(func):
    return tvm.ir.structural_hash(func)

# print(get_dict_key(get_func(50000)))
# print(get_dict_key(fused_fused_decode11_fused_matmul6_cast2))


@T.prim_func(private=True)
def fused_gate_up_decode5_fused_matmul6(lv571: T.Buffer((T.int64(512), T.int64(22016)), "uint32"), lv572: T.Buffer((T.int64(128), T.int64(22016)), "float16"), lv1654: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(22016)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(4096), T.int64(22016)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(22016)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv571[v_i // T.int64(8), v_j], lv572[v_i // T.int64(32), v_j])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv571[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv572[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(22016), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1654[v_i0, v_i1, v_k], p_output0_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1654[v_i0, v_i1, v_k] * p_output0_intermediate[v_k, v_i2]

@T.prim_func(private=True)
def fused_fused_decode10_matmul6(lv571: T.Buffer((T.int64(512), T.int64(22016)), "uint32"), lv572: T.Buffer((T.int64(128), T.int64(22016)), "float16"), lv1654: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(22016)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(4096), T.int64(22016)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(22016)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv571[v_i // T.int64(8), v_j], lv572[v_i // T.int64(32), v_j])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv571[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv572[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(22016), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1654[v_i0, v_i1, v_k], p_output0_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1654[v_i0, v_i1, v_k] * p_output0_intermediate[v_k, v_i2]
            

@T.prim_func(private=True)
def rms_norm1(A: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), B: T.Buffer((T.int64(4096),), "float16"), rms_norm: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    Ared_temp = T.alloc_buffer((T.int64(1), T.int64(1)))
    for bsz, i, k in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("Ared_temp"):
            v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
            T.reads(A[v_bsz, v_i, v_k])
            T.writes(Ared_temp[v_bsz, v_i])
            with T.init():
                Ared_temp[v_bsz, v_i] = T.float32(0)
            Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast("float32", A[v_bsz, v_i, v_k]) * T.Cast("float32", A[v_bsz, v_i, v_k])
    for bsz, i, k in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("rms_norm"):
            v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
            T.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
            T.writes(rms_norm[v_bsz, v_i, v_k])
            rms_norm[v_bsz, v_i, v_k] = T.Cast("float16", T.Cast("float32", B[v_k]) * (T.Cast("float32", A[v_bsz, v_i, v_k]) / T.sqrt(Ared_temp[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))
@T.prim_func(private=True)
def rms_norm2(A: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), B: T.Buffer((T.int64(4096),), "float16"), rms_norm: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    Ared_temp = T.alloc_buffer((T.int64(1), T.int64(1)))
    for bsz, i, k in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("Ared_temp"):
            v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
            T.reads(A[v_bsz, v_i, v_k])
            T.writes(Ared_temp[v_bsz, v_i])
            with T.init():
                Ared_temp[v_bsz, v_i] = T.float32(0)
            Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast("float32", A[v_bsz, v_i, v_k]) * T.Cast("float32", A[v_bsz, v_i, v_k])
    for bsz, i, k in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("rms_norm"):
            v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
            T.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
            T.writes(rms_norm[v_bsz, v_i, v_k])
            rms_norm[v_bsz, v_i, v_k] = T.Cast("float16", T.Cast("float32", B[v_k]) * (T.Cast("float32", A[v_bsz, v_i, v_k]) / T.sqrt(Ared_temp[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))

print(get_dict_key(rms_norm1))
print(get_dict_key(rms_norm2))