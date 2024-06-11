import tvm
from tvm import relay

# 定义待优化的计算图
x = relay.var("x")
y = relay.var("y")
z = relay.add(x, y)
w = relay.subtract(z, relay.const(42))
func = relay.Function([x, y], w)

# 进行操作融合和转换
mod = tvm.IRModule.from_expr(func)
print(mod)
print('================================================================')
# mod = relay.transform.InferType()(mod)
mod = relay.transform.FuseOps(0)(mod)
# mod = relay.transform.PartitionGraph()(mod)

print(mod)
