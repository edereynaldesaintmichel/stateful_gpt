#!/usr/bin/env python
"""
A funny script for evaluating your transformer ablation study.
It loads generated text files from both a vanilla transformer and your novel stateful architecture,
computes the loss of a bigger evaluation model on each text, measures the information content via lzma compression,
and saves the results in JSON format.

Remember: high loss means your text might be more “creative” (or more confused)! Enjoy!
"""

import os
import json
import lzma
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters (from your evaluation script)
batch_size = 256
block_size = 64  # maximum context length for predictions
n_embd = 256
n_head = 8
n_layer = 6
dropout = 0.2
device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

# Load vocabulary from Gutenberg text
with open('combined_gutenberg_10M.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]

# Define the evaluation model components
class Head(nn.Module):
    """One head of self-attention (the snarky kind)."""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention—more heads, more fun!"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """A simple feed-forward block with a twist of ReLU."""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: self-attention meets feed-forward magic."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class EvalModel(nn.Module):
    """The big, burly evaluation model that tells you how real your generated text is."""
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Load the evaluation model and its weights
model = EvalModel().to(device)
model.load_state_dict(torch.load('evaluation_model.pt', map_location=torch.device(device)))
model.eval()

def compute_loss_on_text(text):
    """Compute average loss over blocks for the given text.
    The lower the loss, the more “English-like” your text is (or so we hope!)."""
    tokens = torch.tensor(encode(text), dtype=torch.long, device=device)
    if tokens.numel() < 2:
        return float('nan')
    losses = []
    with torch.no_grad():
        # Break the text into non-overlapping blocks of block_size tokens
        for i in range(0, tokens.size(0) - block_size, block_size//2):
            x = tokens[i:i+block_size].unsqueeze(0)       # shape: (1, block_size)
            y = tokens[i+1:i+block_size+1].unsqueeze(0)     # shape: (1, block_size)
            _, loss = model(x, y)
            losses.append(loss.item())
    if losses:
        return sum(losses) / len(losses)
    else:
        return float('nan')

def compute_information(text):
    """Compute the information metric as the ratio of compressed file size to raw file size.
    Lower ratio means more compressible (i.e., less information) – or just more redundancy."""
    raw_bytes = text.encode('utf-8')
    compressed_bytes = lzma.compress(raw_bytes)
    return len(compressed_bytes) / len(raw_bytes)

# Gather file names for both base and stateful models
base_files = sorted(glob.glob("gen_test/base_temp_*.txt"))
stateful_files = sorted(glob.glob("gen_test/stateful_temp_*_recursion.txt"))
all_files = base_files + stateful_files

results = []

print("Starting evaluation of your mad, genius model outputs...\n")
for filename in all_files:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        loss = compute_loss_on_text(content)
        info = compute_information(content)
        # Extract temperature from filename:
        # e.g. "base_temp_0dot1.txt" -> 0.1, "stateful_temp_0dot1_recursion.txt" -> 0.1
        if filename.startswith("gen_test/base_temp_"):
            temp_str = filename[len("gen_test/base_temp_"):-len(".txt")]
        elif filename.startswith("gen_test/stateful_temp_"):
            temp_str = filename[len("gen_test/stateful_temp_"):-len("_recursion.txt")]
        else:
            temp_str = "0"
        temp = float(temp_str.replace("dot", "."))
        temp = round(1/temp, 2)
        result = {
            "temperature": temp,
            "evaluation_model_loss": loss,
            "information": info,
            "type": "base" if filename.startswith("gen_test/base_temp_") else "stateful",
        }
        results.append(result)
        print(f"Processed {filename}: temperature={temp}, loss={loss:.4f}, information={info:.4f}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Save the results as JSON
with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\nAll done! May your ablation study reveal the secret of the perfect nonsense!")
