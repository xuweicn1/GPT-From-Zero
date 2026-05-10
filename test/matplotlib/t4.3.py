import matplotlib.pyplot as plt
import numpy as np
import torch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def visualize_feature_maps(feature_maps, num_maps=16):
    """可视化特征图输出"""
    if isinstance(feature_maps, torch.Tensor):
        feature_maps = feature_maps.cpu().detach().numpy()
    
    # 假设输入格式: [batch, channels, height, width]
    if len(feature_maps.shape) == 4:
        feature_maps = feature_maps[0]  # 取第一个batch
    
    n_maps = min(num_maps, feature_maps.shape[0])
    n_cols = 4
    n_rows = (n_maps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i in range(n_maps):
        fm = feature_maps[i]
        # 归一化
        fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
        
        axes[i].imshow(fm, cmap='hot')
        axes[i].set_title(f'通道 {i+1}')
        axes[i].axis('off')
    
    for i in range(n_maps, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('特征图可视化', fontsize=14)
    plt.tight_layout()
    plt.show()
    
# 创建一个简单的特征图用于测试
test_feature_maps = torch.randn(32, 28, 28)  # 32个通道，尺寸 28x28
visualize_feature_maps(test_feature_maps, num_maps=16)
