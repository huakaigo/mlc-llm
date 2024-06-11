import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()

# Iterate over the named parameters of the network
for name, param in net.named_parameters():
    print(name, param.size())

net.eval()
input = torch.randn(1,3,32,32)
output = net(input)
print("---------------run-----------------")
print(output.shape)
print(f"result is : {output}")

print("---------------save-----------------")
torch.save(net.state_dict(), "test.pth")
print("---------------load run-----------------")

new_net = Net()
new_net.load_state_dict(torch.load('test.pth'))

new_net.eval()
new_output = new_net(input)
print("---------------run-----------------")
print(new_output.shape)
print(f"result is : {new_output}")
