import matplotlib.pyplot as plt

# 设置字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 测试
plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3], [1, 4, 9])
plt.xlabel('横坐标 (X轴)')
plt.ylabel('纵坐标 (Y轴)')
plt.title('中文标题测试 - 正常显示')
plt.show()