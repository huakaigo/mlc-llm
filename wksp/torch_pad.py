import torch

# 假设您有一个 shape 为 [A, B] 的 Tensor
A, B = 5, 6
input_tensor = torch.randn(A, B)

# 将 shape 变换为 [A+3, B] 并进行零填充
padding = (0, 0, 3, 0)  # 分别表示左、右、上、下填充的数目
output_tensor = torch.nn.functional.pad(input_tensor, padding)

print(output_tensor.shape)  # 输出为 [A+3, B]
