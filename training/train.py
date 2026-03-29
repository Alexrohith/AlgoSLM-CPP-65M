import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from model.gpt_model import AlgoSLM
from training.dataset import CPDataset


# =====================
# TRAINING CONFIG
# =====================
DEVICE = "cuda"
SEQ_LEN = 512        # increase context
BATCH_SIZE = 4       # RTX 3050 should handle this
EPOCHS = 3
LR = 2e-4


def train():

    print("Loading dataset...")
    dataset = CPDataset(
        file_path="data/cleaned/general_cpp/general_cpp.txt",
        tokenizer_path="tokenizer/cpp_tokenizer.json",
        seq_len=SEQ_LEN
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    print("Initializing model...")
    model = AlgoSLM().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    scaler = GradScaler("cuda")

    # Create checkpoint directory safely
    os.makedirs("checkpoints", exist_ok=True)

    model.train()

    print("Starting training...\n")

    for epoch in range(EPOCHS):

        total_loss = 0

        for step, (x, y) in enumerate(loader):

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            with autocast("cuda"):
                logits = model(x)
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"\nEpoch {epoch} completed. Avg Loss: {avg_loss:.4f}\n")

        # Save checkpoint
        torch.save(
            model.state_dict(),
            f"checkpoints/model_epoch_{epoch}.pt"
        )

        print(f"Checkpoint saved: checkpoints/model_epoch_{epoch}.pt\n")


if __name__ == "__main__":
    train()