from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer/cpp_tokenizer.json")

with open("data/cleaned/cp_finetune.txt","r",encoding="utf-8") as f:
    raw = f.read()

tokens = tok.encode(raw).ids
print("Total tokens:", len(tokens))