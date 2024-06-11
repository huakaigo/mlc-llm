import tvm
from tvm import relay
import onnx

# 加载 ONNX 模型
onnx_model = onnx.load("resnet50-v2-7.onnx")

# 定义输入形状
shape_dict = {"data": (1, 3, 224, 224)}  # 以字典形式定义输入的名称和形状

# 将 ONNX 模型转换为 Relay 模块
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# 打印模块和参数信息
print("Module:", mod)
print("Parameters:", params)
