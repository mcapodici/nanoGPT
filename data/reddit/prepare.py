import os
import requests
import tiktoken
import numpy as np

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
train_file_path = os.path.join(os.path.dirname(__file__), 'train.bin')
val_file_path = os.path.join(os.path.dirname(__file__), 'val.bin')

if (os.path.exists(train_file_path)):
    print("Training file exists, quitting")
    exit(0)

with open(input_file_path, 'r') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(train_file_path)
val_ids.tofile(val_file_path)

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
