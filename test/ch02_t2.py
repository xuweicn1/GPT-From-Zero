import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 创建一个小型 embedding
vocab_size = 10   # 10个词
embed_dim = 2     # 2维向量（方便画图）
embedding = nn.Embedding(vocab_size, embed_dim)

# 获取所有权重
weights = embedding.weight.data.numpy()

# 画图
plt.figure(figsize=(8, 6))
plt.scatter(weights[:, 0], weights[:, 1], s=100)
for i, (x, y) in enumerate(weights):
    plt.annotate(str(i), (x, y), fontsize=12, ha='center')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Token Embeddings Visualization (2D)')
plt.grid(True, alpha=0.3)
plt.show()

# 查询相似的词
token_a = torch.tensor([0])
token_b = torch.tensor([1])

vec_a = embedding(token_a)
vec_b = embedding(token_b)

# 计算余弦相似度
similarity = torch.cosine_similarity(vec_a, vec_b)
print(f"Token 0 和 Token 1 的相似度: {similarity.item():.4f}")