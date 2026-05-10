import matplotlib.pyplot as plt
import numpy as np
import torch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class LivePlotter:
    def __init__(self):
        plt.ion()  # 交互模式
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 5))
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def update(self, epoch, train_loss, val_loss, train_acc, val_acc):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        
        # 清空并重绘
        self.ax1.clear()
        self.ax2.clear()
        
        # 损失图
        self.ax1.plot(self.train_losses, 'b-', label='训练损失', linewidth=2)
        self.ax1.plot(self.val_losses, 'r-', label='验证损失', linewidth=2)
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('损失曲线')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # 准确率图
        self.ax2.plot(self.train_accs, 'b-', label='训练准确率', linewidth=2)
        self.ax2.plot(self.val_accs, 'r-', label='验证准确率', linewidth=2)
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy (%)')
        self.ax2.set_title('准确率曲线')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Epoch {epoch}', fontsize=12)
        plt.pause(0.01)
    
    def close(self):
        plt.ioff()
        plt.show()

# 使用示例
plotter = LivePlotter()
for epoch in range(1, 51):
    # 模拟训练过程
    train_loss = 2 / np.sqrt(epoch) + np.random.normal(0, 0.05)
    val_loss = 2.2 / np.sqrt(epoch) + np.random.normal(0, 0.05)
    train_acc = 100 * (1 - np.exp(-epoch/20))
    val_acc = 100 * (1 - np.exp(-epoch/25))
    
    plotter.update(epoch, train_loss, val_loss, train_acc, val_acc)

plotter.close()