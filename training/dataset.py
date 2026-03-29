import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class CPDataset(Dataset):
    def __init__(self, file_path, tokenizer_path, seq_len=1024):
        self.seq_len = seq_len

        print("Loading tokenizer...")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        print("Reading dataset...")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        print("Tokenizing dataset...")
        tokens = self.tokenizer.encode(text).ids

        self.tokens = torch.tensor(tokens, dtype=torch.long)

        print(f"Total tokens: {len(self.tokens)}")

    def __len__(self):
        return len(self.tokens) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1

        chunk = self.tokens[start:end]

        x = chunk[:-1]
        y = chunk[1:]

        return x, y