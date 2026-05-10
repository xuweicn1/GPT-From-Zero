import matplotlib.pyplot as plt
import torch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def visualize_filters(model, layer_name='conv1', num_filters=16):
    """可视化卷积层过滤器"""
    # 获取指定层的权重
    for name, param in model.named_parameters():
        if layer_name in name and 'weight' in name:
            filters = param.data.cpu().numpy()
            break
    
    # 限制显示数量
    n_filters = min(num_filters, filters.shape[0])
    
    # 计算子图布局
    n_cols = 4
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i in range(n_filters):
        # 归一化以便显示
        filter_img = filters[i, 0]  # 第一个通道
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())
        
        axes[i].imshow(filter_img, cmap='viridis')
        axes[i].set_title(f'Filter {i+1}')
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(n_filters, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'卷积层 {layer_name} 可视化', fontsize=14)
    plt.tight_layout()
    plt.show()



# 创建最简单的模型
model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3))

# 调用可视化（需要先定义 visualize_filters 函数）
visualize_filters(model, '0', 8)  # Sequential 的层名是 '0'