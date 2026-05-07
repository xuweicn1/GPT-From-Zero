import torch
import torchvision

# 从列表创建
data = [[1, 2], [3, 4]]
tensor = torch.tensor(data)

# 创建全0、全1或随机张量
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
random = torch.rand(2, 3)  # 均匀分布

print(f"张量形状: {tensor.shape}")
print(f"数据类型: {tensor.dtype}")
print(f"所在设备: {tensor.device}")  # cpu 或 cuda:0



# 需要求导的张量
x = torch.tensor([2.0], requires_grad=True)
# 进行操作
y = x ** 2 + 3 * x + 1
# 反向传播，自动计算梯度 dy/dx
y.backward()

# 打印梯度
print(x.grad)  # 输出: tensor([7.]) 因为 2*x + 3 = 7



import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(input_size, hidden_size)   # 全连接层1
        self.relu = nn.ReLU()                           # 激活函数
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 全连接层2 (输出层)

    # 定义数据如何向前传播
    def forward(self, x):
        out = self.fc1(x)      # x -> fc1
        out = self.relu(out)   # 应用激活函数
        out = self.fc2(out)    # 得到最终输出
        return out

# 实例化网络
model = SimpleNet(input_size=784, hidden_size=128, num_classes=10)
print(model)




import os
print("当前工作目录:", os.getcwd())
print("data/MNIST/raw 是否存在:", os.path.exists('./data/MNIST/raw'))
print("训练集文件是否存在:", os.path.exists('./data/MNIST/raw/train-images-idx3-ubyte.gz'))


import torch.optim as optim

# --- 1. 准备数据 (以MNIST手写数字为例) ---
from torchvision import datasets, transforms


# datasets.MNIST.mirrors = ['https://ossci-datasets.s3.amazonaws.com/mnist/']
# datasets.MNIST.mirrors = ['http://yann.lecun.com/exdb/mnist/']

# 数据预处理：转为Tensor并归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载并加载MNIST训练集
# trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# download=True 会自动检测文件是否存在，存在就跳过下载
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)



# --- 2. 初始化模型、损失函数和优化器 ---
# model = SimpleNet(input_size=784, hidden_size=128, num_classes=10)
# criterion = nn.CrossEntropyLoss()           # 分类问题用交叉熵损失
# optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

# # --- 3. 训练循环 ---
# epochs = 5
# for epoch in range(epochs):
#     running_loss = 0.0
#     for images, labels in trainloader:
#         # 将28x28的图片展平成784维的向量
#         images = images.view(images.size(0), -1)

#         # 梯度清零 (否则梯度会累积)
#         optimizer.zero_grad()

#         # 前向传播、计算损失、反向传播、更新权重
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
    
#     print(f'Epoch {epoch+1}, Loss: {running_loss / len(trainloader):.4f}')

# print("训练完成！")


