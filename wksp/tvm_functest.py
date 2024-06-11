import tvm
from tvm import relax, te
from tvm.script import relax as R
from tvm.relax.expr import Expr, Function, Var
from tvm.relax.testing import nn

# 创建一个简单的神经网络模型
x = relax.Var('x', relax.TensorStructInfo((10, 10)))
y = nn.Linear(10, 5, False)
z = nn.ReLU()
f = te.compute((10, 5), fcompute = lambda i: z(y(x[i])), name = "easy_net")
# model = relax.Function(relax.analysis.free_vars(z), z)

# 创建nn.Module对象
module = tvm.IRModule()
module["main"] = f

# 访问param_info字段
param_info = module["main"].param_info

# 打印参数信息
for param_name, info in param_info.items():
    print(f"Parameter: {param_name}")
    print(f"Shape: {info['shape']}")
    print(f"Dtype: {info['dtype']}")
    print(f"Initialization: {info['init']}")
    print(f"Storage ID: {info['storage_id']}")
    print()
