import os
import re
import urllib.request


class SimpleTokenizerV1:
    def __init__(self, vocab=None):
        """
        vocab: 字符串到整数的映射，例如 {'A': 11, 'Ah': 12, 'Among': 13}

        """
        self.str_to_init = vocab
        # 反转字典，得到 init_to_str 示例： {11: 'A', 12: 'Ah', 13: 'Among'}
        self.init_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        """
        将输入文本转换为整数列表
        """
        prosessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        prosessed = [item.strip() for item in prosessed if item.strip()]
        ids = [self.str_to_init[s] for s in prosessed]
        return ids

    def decode(self, ids):
        """
        将整数列表转换回文本
        """

        text = " ".join([self.init_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)

        return text


url = "https://raw.githubusercontent.com/xuweicn1/GPT-From-Zero/refs/heads/ch02/ch02/the-verdict.txt"
file_path = "the-verdict.txt"

if not os.path.exists("the-verdict.txt"):
    urllib.request.urlretrieve(url, file_path)

with open(file_path, "r") as f:
    raw_text = f.read()


prosessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
prosessed = [item.strip() for item in prosessed if item.strip()]
all_words = sorted(set(prosessed))
vocab = {token: inter for inter, token in enumerate(all_words)}

# print(vocab)


tokenzier = SimpleTokenizerV1(vocab)

# text = "Among the verdicts, the most controversial was the one that acquitted the defendant."

text = """ It's the last he painted,you know,"
Mrs. Gisburn said with pardonable pride. """


ids = tokenzier.encode(text)
print(ids)


print(tokenzier.decode(ids))

text = "Hello, do you like tea?"

print(tokenzier.encode(text))

# print(tokenzier.init_to_str)
# print(tokenzier.str_to_init)
