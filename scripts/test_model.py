import torch
from model.gpt_model import AlgoSLM

model = AlgoSLM().cuda()

x = torch.randint(0, 20000, (2, 128)).cuda()
logits = model(x)

print("Output shape:", logits.shape)

total_params = sum(p.numel() for p in model.parameters())
print("Total parameters:", total_params / 1e6, "Million")