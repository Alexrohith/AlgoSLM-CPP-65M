import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from model.gpt_model import AlgoSLM
from training.dataset import CPDataset


DEVICE = "cuda"
SEQ_LEN = 512

# small dataset → use batch size 1
BATCH_SIZE = 2

# alignment step → only 1 epoch
EPOCHS = 6

# very small LR for alignment
LR = 3e-5


def find_reasoning_dataset():

    possible_paths = [
        "data/cleaned/cp_reasoning_finetune.txt",
        "data/cleaned/general_cpp/cp_reasoning_finetune.txt",
        "training/data/cleaned/cp_reasoning_finetune.txt"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found reasoning dataset at: {path}")
            return path

    raise FileNotFoundError(
        "cp_reasoning_finetune.txt not found."
    )


def finetune():

    print("=== Reasoning Alignment Fine-Tuning ===\n")

    dataset_path = find_reasoning_dataset()

    print("Loading reasoning dataset...")

    dataset = CPDataset(
        file_path=dataset_path,
        tokenizer_path="tokenizer/cpp_tokenizer.json",
        seq_len=SEQ_LEN
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False   # IMPORTANT for small dataset
    )

    print("Loading pretrained CP model...")

    model = AlgoSLM().to(DEVICE)

    model.load_state_dict(
        torch.load(
            "checkpoints/model_finetuned_epoch_0.pt",
            weights_only=True
        )
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    criterion = nn.CrossEntropyLoss()

    scaler = GradScaler("cuda")

    os.makedirs("checkpoints", exist_ok=True)

    model.train()

    print("\nStarting reasoning fine-tuning...\n")

    for epoch in range(EPOCHS):

        total_loss = 0
        step_count = 0

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
            step_count += 1

            print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

        if step_count == 0:
            print("Dataset too small to train.")
            return

        avg_loss = total_loss / step_count

        print(f"\nEpoch {epoch} completed.")
        print(f"Average Loss: {avg_loss:.4f}\n")

        checkpoint_path = f"checkpoints/model_reasoning_epoch_{epoch}.pt"

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        print(f"Saved checkpoint: {checkpoint_path}\n")


if __name__ == "__main__":
    finetune()