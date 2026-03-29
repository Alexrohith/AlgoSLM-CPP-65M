import torch
import torch.nn as nn
from model.transformer_block import TransformerBlock
from configs.model_config import MODEL_CONFIG


class AlgoSLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.vocab_size = MODEL_CONFIG["vocab_size"]
        self.max_seq_len = MODEL_CONFIG["max_seq_len"]
        self.n_layers = MODEL_CONFIG["n_layers"]
        self.embed_dim = MODEL_CONFIG["embed_dim"]

        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embed_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.embed_dim,
                n_heads=MODEL_CONFIG["n_heads"],
                ffn_dim=MODEL_CONFIG["ffn_dim"],
                dropout=MODEL_CONFIG["dropout"],
            )
            for _ in range(self.n_layers)
        ])

        self.ln_f = nn.LayerNorm(self.embed_dim)

        self.output_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)

        # Weight tying
        self.output_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        B, T = input_ids.size()
        assert T <= self.max_seq_len

        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)

        logits = self.output_head(x)
        return logits