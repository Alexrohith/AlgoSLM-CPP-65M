from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import Sequence, NFD, StripAccents
import os

files = [
    "data/cleaned/general_cpp/general_cpp.txt",
    "data/cleaned/cp_pairs/cp_solutions.txt"
]

os.makedirs("tokenizer", exist_ok=True)

tokenizer = Tokenizer(BPE(unk_token="<unk>"))

tokenizer.normalizer = Sequence([NFD(), StripAccents()])
tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(
    vocab_size=20000,
    special_tokens=[
        "<unk>",
        "<bos>",
        "<eos>",
        "<problem>",
        "<code>"
    ]
)

tokenizer.train(files, trainer)

tokenizer.save("tokenizer/cpp_tokenizer.json")

print("Tokenizer training complete.")