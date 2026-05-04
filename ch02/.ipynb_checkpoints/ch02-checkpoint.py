import os

import urllib.request  

url ='https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt'
file_path = 'the-verdict.txt'

if not os.path.exists('the-verdict.txt'):
    urllib.request.urlretrieve(url, file_path)

