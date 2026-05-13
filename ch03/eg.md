好的，我们继续把第2章也改编成适合Python初学者的教程。这一章讲的是**文本数据处理**——也就是如何把原始文字，一步步变成大语言模型能“吃”的输入。

---

## 第二章：给大模型准备“食物”——文本数据处理

在上一章，我们学会了注意力机制的原理。但你是否想过：文字是怎么输入到模型里的？模型可不认识“苹果”这两个字，它只懂数字。

本章就带你一步步把一段文本，变成模型能理解的数字形式。就像做饭，我们要把原材料（原始文本）经过清洗、切割、调味（分词、编码、嵌入），最终做成一道模型爱吃的“数字大餐”。

### 2.1 准备工作：导入工具并下载数据

首先，导入我们需要用到的库，并下载一本公共领域的短篇小说《The Verdict》作为示例文本。


```python
# 导入需要的库
import os
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
from importlib.metadata import version

print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))

# 下载示例文本（如果本地没有的话）
if not os.path.exists("the-verdict.txt"):
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()  # 如果下载失败会抛出异常
    with open("the-verdict.txt", "wb") as f:
        f.write(response.content)
    print("小说下载完成!")
else:
    print("小说文件已存在!")
```

    torch version: 2.4.0
    tiktoken version: 0.7.0
    小说下载完成!


### 2.2 看一眼我们要处理的数据

让我们读取文本，看看里面有什么。


```python
# 读取文本文件
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"总字符数: {len(raw_text)}")
print(f"\n开头100个字符:\n{raw_text[:100]}")
```

    总字符数: 20479
    
    开头100个字符:
    I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no 


---

### 2.3 第一步：分词（Tokenization）——把句子切成“词块”

模型不能直接理解一整个句子，需要把句子切成一个个“词块”（token）。最简单的分词方式是按空格和标点符号来切分。

#### 2.3.1 一个最基础的分词实验

我们先写一个简单的正则表达式来试着分词。


```python
import re

# 测试文本
text = "Hello, world. Is this-- a test?"
print(f"原始文本: {text}\n")

# 第一步：按标点和空格切分
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
print(f"切分后: {result}")

# 第二步：去掉空白和多余的空字符串
result = [item.strip() for item in result if item.strip()]
print(f"\n清理后: {result}")
```

    原始文本: Hello, world. Is this-- a test?
    
    切分后: ['Hello', ',', '', ' ', 'world', '.', '', ' ', 'Is', ' ', 'this', '--', '', ' ', 'a', ' ', 'test', '?', '']
    
    清理后: ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']


**代码解释**：
- `re.split()` 是正则表达式分割函数
- 括号 `()` 表示保留分隔符本身
- `\s` 表示空白字符（空格、换行等）
- `|` 表示“或”
- 最后用列表推导式过滤掉空字符串

#### 2.3.2 对整个小说分词

现在把同样的方法应用到我们下载的小说上。


```python
# 对整个文本进行分词
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

print(f"总词块数: {len(preprocessed)}")
print(f"前30个词块: {preprocessed[:30]}")
```

    总词块数: 4690
    前30个词块: ['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 'genius', '--', 'though', 'a', 'good', 'fellow', 'enough', '--', 'so', 'it', 'was', 'no', 'great', 'surprise', 'to', 'me', 'to', 'hear', 'that', ',', 'in']


---

### 2.4 第二步：构建词表（Vocabulary）——给每个词编号

现在我们有了一堆词块，但模型只懂数字。我们需要给每个独特的词块分配一个唯一的数字ID。

#### 2.4.1 创建词表


```python
# 获取所有不重复的词块，排序后分配ID
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(f"词表大小: {vocab_size}")

# 创建 词→ID 的映射字典
vocab = {token: integer for integer, token in enumerate(all_words)}

# 查看前10个映射
for i, (word, idx) in enumerate(vocab.items()):
    if i >= 10:
        break
    print(f"'{word}' -> {idx}")
```

    词表大小: 1130
    '!' -> 0
    '"' -> 1
    ''' -> 2
    '(' -> 3
    ')' -> 4
    ',' -> 5
    '--' -> 6
    '.' -> 7
    ':' -> 8
    ';' -> 9


> **知识点**：词表就是一本“词典”，模型通过查词典把文字变成数字。每个独特的词都有一个唯一的编号。

#### 2.4.2 实现一个简单的分词器类

我们把编码（文字→数字）和解码（数字→文字）功能封装成一个类。


```python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab                    # 文字→数字
        self.int_to_str = {i: s for s, i in vocab.items()}  # 数字→文字（反查表）

    def encode(self, text):
        """把文字转换成数字列表"""
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        """把数字列表还原成文字"""
        text = " ".join([self.int_to_str[i] for i in ids])
        # 把标点前的空格去掉，比如 "world ." → "world."
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# 测试我们的分词器
tokenizer = SimpleTokenizerV1(vocab)

text = '''"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.'''
ids = tokenizer.encode(text)
print(f"编码结果: {ids}")
print(f"解码验证: {tokenizer.decode(ids)}")
```

    编码结果: [1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]
    解码验证: " It' s the last he painted, you know," Mrs. Gisburn said with pardonable pride.


---

### 2.5 第三步：处理生词——添加特殊标记

我们现在的分词器有一个致命问题：如果遇到不认识的词，它会直接报错。


```python
# 尝试编码一个新句子，其中 "Hello" 不在词表中
tokenizer = SimpleTokenizerV1(vocab)
text = "Hello, do you like tea?"

try:
    tokenizer.encode(text)
except KeyError as e:
    print(f"错误！不认识这个词: {e}")
```

    错误！不认识这个词: 'Hello'


为了解决这个问题，我们需要在词表中加入两个特殊标记：
- **`<|unk|>`**：表示不认识的词
- **`<|endoftext|>`**：表示文本结束（GPT-2 用它来分隔不同的文档）


```python
# 扩展词表，加入特殊标记
all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}

print(f"新词表大小: {len(vocab)}")
print(f"最后两个标记: {list(vocab.items())[-2:]}")
```

    新词表大小: 1132
    最后两个标记: [('<|endoftext|>', 1130), ('<|unk|>', 1131)]


现在写一个升级版的分词器，遇到不认识的词就自动替换成 `<|unk|>`：


```python
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # 关键改进：遇到不认识的词就换成 <|unk|>
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# 测试升级版分词器
tokenizer = SimpleTokenizerV2(vocab)

# 用 <|endoftext|> 连接两段不同的文本
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
combined = " <|endoftext|> ".join((text1, text2))

print(f"合并文本: {combined}")
print(f"编码结果: {tokenizer.encode(combined)}")
print(f"解码结果: {tokenizer.decode(tokenizer.encode(combined))}")
```

    合并文本: Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.
    编码结果: [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]
    解码结果: <|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.


可以看到，`Hello` 和 `palace` 因为在原小说中没出现过，都被替换成了 `<|unk|>`。

---

### 2.6 第四步：使用工业级分词器（BPE）

我们自己写的分词器太简陋了。真实的大模型（如GPT-2）使用的是**字节对编码（BPE）**，它能智能地把生词拆成更小的子词单元。我们直接使用OpenAI的 `tiktoken` 库。


```python
# 加载 GPT-2 的分词器
tokenizer = tiktoken.get_encoding("gpt2")

# 测试：把 "someunknownPlace" 这个生造词进行分词
text = "Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknownPlace."
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(f"编码结果: {integers}")
print(f"解码验证: {tokenizer.decode(integers)}")
```

    编码结果: [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 1659, 617, 34680, 27271, 13]
    解码验证: Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknownPlace.


> **BPE的魔力**：注意 `someunknownPlace` 被自动拆成了 `some` + `unknown` + `Place` 三个子词（对应ID `617, 34680, 27271`）。这就是BPE的厉害之处——即使遇到从没见过的词，它也能通过子词组合来理解。

---

### 2.7 第五步：制作训练样本——滑动窗口

语言模型的任务是“预测下一个词”。所以我们要把长文本切成很多小段，每段包含“上文（输入）”和“下文（要预测的目标）”。


```python
# 把整本小说用 GPT-2 分词器编码
enc_text = tokenizer.encode(raw_text)
print(f"编码后总长度: {len(enc_text)}")

# 演示：上下文窗口为4时，输入和目标的关系
enc_sample = enc_text[50:]  # 跳过开头
context_size = 4

x = enc_sample[:context_size]          # 输入：前4个词
y = enc_sample[1:context_size + 1]     # 目标：后4个词（就是输入往右移一位）

print(f"输入 (x): {x}")
print(f"目标 (y):      {y}")
print(f"\n对应文字:")
print(f"输入:  {tokenizer.decode(x)}")
print(f"目标:      {tokenizer.decode(y)}")
```

    编码后总长度: 5145
    输入 (x): [290, 4920, 2241, 287]
    目标 (y):      [4920, 2241, 287, 257]
    
    对应文字:
    输入:   and established himself in
    目标:       established himself in a


**关系图解**：
```
输入:  [290, 4920, 2241, 287]   →   目标:  [4920, 2241, 287, 257]
        ↓    ↓     ↓     ↓              ↓    ↓     ↓     ↓
       and  estab... him..  in         estab... him..  in    a
```
目标就是输入整体往右移动一个位置，这样模型就能学会：给定 `[290]` 预测 `4920`，给定 `[290, 4920]` 预测 `2241`，以此类推。

---

### 2.8 第六步：封装成数据集和数据加载器

用 PyTorch 的 `Dataset` 和 `DataLoader` 来批量生成训练样本，这会是我们后续训练 GPT 模型的基础设施。


```python
class GPTDatasetV1(Dataset):
    """GPT 训练数据集：用滑动窗口把文本切成 (输入, 目标) 对"""
    
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 对整个文本进行编码
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 用滑动窗口切分
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    """创建一个数据加载器的快捷函数"""
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,  # 丢弃最后不够一个batch的数据
        num_workers=num_workers
    )
    return dataloader


# 测试：batch_size=8, 上下文长度=4, 步长=4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("输入批次 (Inputs):")
print(inputs)
print(f"\n目标批次 (Targets):")
print(targets)
```

    输入批次 (Inputs):
    tensor([[   40,   367,  2885,  1464],
            [ 1807,  3619,   402,   271],
            [10899,  2138,   257,  7026],
            [15632,   438,  2016,   257],
            [  922,  5891,  1576,   438],
            [  568,   340,   373,   645],
            [ 1049,  5975,   284,   502],
            [  284,  3285,   326,    11]])
    
    目标批次 (Targets):
    tensor([[  367,  2885,  1464,  1807],
            [ 3619,   402,   271, 10899],
            [ 2138,   257,  7026, 15632],
            [  438,  2016,   257,   922],
            [ 5891,  1576,   438,   568],
            [  340,   373,   645,  1049],
            [ 5975,   284,   502,   284],
            [ 3285,   326,    11,   287]])


---

### 2.9 第七步：词嵌入（Token Embedding）——把数字变成向量

现在每个词都有了数字ID，但数字本身不能表达语义。我们需要把每个ID映射成一个**密集向量**，这就是“嵌入”。


```python
# 假设一个迷你词表（6个词），想把每个词映射成3维向量
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = nn.Embedding(vocab_size, output_dim)

print("嵌入层的权重矩阵（6行3列）:")
print(embedding_layer.weight)
print(f"\n形状: {embedding_layer.weight.shape}")
```

    嵌入层的权重矩阵（6行3列）:
    Parameter containing:
    tensor([[ 0.3374, -0.1778, -0.1690],
            [ 0.9178,  1.5810,  1.3010],
            [ 1.2753, -0.2010, -0.1606],
            [-0.4015,  0.9666, -1.1481],
            [-1.1589,  0.3255, -0.6315],
            [-2.8400, -0.7849, -1.4096]], requires_grad=True)
    
    形状: torch.Size([6, 3])


**工作原理**：嵌入层就是一个查表操作。输入一个词ID，它就返回权重矩阵中对应的那一行。


```python
# 查表演示：获取ID为3的词向量
print(f"ID=3 的向量: {embedding_layer(torch.tensor([3]))}")

# 批处理：一次查4个ID
input_ids = torch.tensor([2, 3, 5, 1])
print(f"\n批量查询四个ID:\n{embedding_layer(input_ids)}")
```

    ID=3 的向量: tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)
    
    批量查询四个ID:
    tensor([[ 1.2753, -0.2010, -0.1606],
            [-0.4015,  0.9666, -1.1481],
            [-2.8400, -0.7849, -1.4096],
            [ 0.9178,  1.5810,  1.3010]], grad_fn=<EmbeddingBackward0>)


---

### 2.10 第八步：位置编码（Positional Encoding）——告诉模型词的位置

“我爱你”和“你爱我”包含同样的三个词，但意思完全不同。嵌入层本身无法区分位置，所以我们需要**位置编码**来注入位置信息。

GPT-2 使用**可学习的绝对位置嵌入**。简单来说，就是给位置0、位置1、位置2……也各分配一个向量，然后加到词向量上。


```python
# 创建位置嵌入层（假设上下文长度为4，嵌入维度256）
context_length = 4
output_dim = 256

pos_embedding_layer = nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))  # 位置0,1,2,3

print(f"位置嵌入形状: {pos_embeddings.shape}")
print(f"位置0的向量（前5个值）: {pos_embeddings[0][:5]}")
```

    位置嵌入形状: torch.Size([4, 256])
    位置0的向量（前5个值）: tensor([-0.2951, -2.6120, -0.1683,  0.0064, -0.0581], grad_fn=<SliceBackward0>)


#### 完整的输入嵌入 = 词嵌入 + 位置嵌入


```python
# 用真实的 GPT-2 词表大小创建嵌入层
vocab_size = 50257  # GPT-2 的词表大小
output_dim = 256

token_embedding_layer = nn.Embedding(vocab_size, output_dim)

# 从数据加载器取一批数据
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# 获取词嵌入
token_embeddings = token_embedding_layer(inputs)
print(f"词嵌入形状: {token_embeddings.shape}")

# 获取位置嵌入（位置0到3）
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(f"位置嵌入形状: {pos_embeddings.shape}")

# 最终输入嵌入 = 词嵌入 + 位置嵌入
input_embeddings = token_embeddings + pos_embeddings
print(f"最终输入嵌入形状: {input_embeddings.shape}")
```

    词嵌入形状: torch.Size([8, 4, 256])
    位置嵌入形状: torch.Size([4, 256])
    最终输入嵌入形状: torch.Size([8, 4, 256])


最终得到的 `[8, 4, 256]` 张量，就是8个样本、每个样本4个词、每个词用256维向量表示的“模型级”输入。这个输入就可以直接送给我们第三章学的注意力机制了！

---

### 本章总结

恭喜你！我们走完了从原始文本到模型输入的全流程：

| 步骤 | 做什么 | 输入 | 输出 |
|:---|:---|:---|:---|
| **分词** | 把句子切成词块 | `"Hello, world."` | `["Hello", ",", "world", "."]` |
| **编码** | 把词块映射成ID | `["Hello", ",", "world", "."]` | `[15496, 11, 995, 13]` |
| **嵌入** | 把ID映射成向量 | `[15496, 11, 995, 13]` | `4个 256维向量` |
| **位置编码** | 加入位置信息 | 位置0,1,2,3 | `4个 256维向量` |
| **相加** | 得到最终输入 | 词向量+位置向量 | `[4, 256]` 张量 |

这个管道就是 GPT 模型的“消化系统”。下一章，我们将把这个输入送进注意力机制，让模型开始真正“理解”语言！