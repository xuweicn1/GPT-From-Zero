import matplotlib.pyplot as plt
import numpy as np
import torch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False



# 模拟训练过程
epochs = range(1, 51)
train_loss = np.exp(-np.array(epochs)/10) + np.random.normal(0, 0.02, 50)
val_loss = np.exp(-np.array(epochs)/12) + np.random.normal(0, 0.02, 50)

# plt.figure(figsize=(10, 6))
# plt.plot(epochs, train_loss, 'b-', label='train', linewidth=2)
# plt.plot(epochs, val_loss, 'r-', label='loss', linewidth=2)
# plt.xlabel('Epoch', fontsize=12)
# plt.ylabel('Loss', fontsize=12)
# plt.title('loss ', fontsize=14)
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()


plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'b-', label='训练损失', linewidth=2)
plt.plot(epochs, val_loss, 'r-', label='验证损失', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('训练过程损失曲线', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()