from tvm.script import ir as I
from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(A: R.Tensor((3, 4), dtype="float32"), B: R.Tensor((4, 5), dtype="float32")):
        with R.dataflow():
            lv: R.Tensor((3, 5), dtype="float32") = R.matmul(A, B)
            gv: R.Tensor((3, 5), dtype="float32") = lv
            R.output(gv)
        return gv