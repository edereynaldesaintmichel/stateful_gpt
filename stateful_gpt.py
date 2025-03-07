import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# hyperparameters
batch_size = 2048         # number of sequences processed in parallel
block_size = 33           # maximum context length for predictions
max_epochs = 40         # total number of epochs to train
learning_rate = 3e-4 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
RECURSION_STEPS = 0      # number of additional recursion steps
exponential_base = 2      # loss multiplier for recursion steps
n_embd = 64
n_head = 8
n_layer = 8
dropout = 0.2
contribution_coeff = 1

# ----------------------------
# Load data and build vocabulary
with open('combined_gutenberg_10M.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ----------------------------
# Create a dataset that splits data into contiguous chunks.
# Each sample consists of block_size tokens as input and the next block_size tokens as targets.
class TextDataset(Dataset):
    def __init__(self, data, block_size):
        # Use non-overlapping chunks (adjust stride if you prefer overlap)
        self.data = data
        self.block_size = block_size
        self.num_samples = (len(data) - 1) // block_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Calculate start index for this chunk
        i = idx * self.block_size
        x = self.data[i: i + self.block_size]
        y = self.data[i + 1: i + self.block_size + 1]
        return x, y

train_dataset = TextDataset(train_data, block_size)
val_dataset = TextDataset(val_data, block_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ----------------------------
# Model definition (unchanged)
class MultiHeadAttention(nn.Module):
    """Using PyTorch's optimized MultiheadAttention implementation."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        mask = self.mask[:T, :T] == 0
        attn_output, _ = self.attn(x, x, x, attn_mask=mask, is_causal=True, need_weights=False)
        return self.dropout(self.proj(attn_output))

class FeedForward(nn.Module):
    """Simple feed-forward network."""
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
    """Transformer block with optimized attention."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.hidden_state_value = nn.Sequential(
            nn.Linear(n_embd, n_embd)
        )
        self.hidden_state_key = nn.Sequential(
            nn.Linear(n_embd, n_embd)
        )
        self.relu = nn.ReLU()
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module is self.hidden_state_key or module in self.hidden_state_value:
                return
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    # def get_recursion_step_encoding(self, recursion_step, batch_size, seq_length):
    #     encoding_dim = n_embd // 8
    #     recursion_pos = torch.tensor(recursion_step, device=device, dtype=torch.float)
    #     dim_indices = torch.arange(0, encoding_dim, device=device).float()
    #     frequencies = 1.0 / torch.pow(2*torch.pi*10, (2 * (dim_indices // 2)) / encoding_dim)
    #     pos_encodings = recursion_pos * frequencies
    #     encoding = torch.zeros(encoding_dim, device=device)
    #     encoding[0::2] = torch.sin(pos_encodings[0::2])
    #     encoding[1::2] = torch.cos(pos_encodings[1::2])
        
    #     encoding = encoding.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, encoding_dim]
    #     encoding = encoding.expand(batch_size, seq_length, -1)  # Shape: [B, T, encoding_dim]
        
    #     return encoding
    
    def forward(self, idx, targets=None, prev_hidden_states=None, current_recursion_step=0):
        B, T = idx.shape
        # Token embeddings
        tok_emb = self.token_embedding_table(idx)
        # Position embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        
        # If previous hidden states are provided, process and add them
        if prev_hidden_states is not None:
            hidden_states_values = torch.zeros_like(prev_hidden_states)
            hidden_states_keys = torch.zeros_like(prev_hidden_states)
            hidden_states_values[:, 1:] = self.hidden_state_value(prev_hidden_states[:, :-1])
            prev_hidden_state_values = hidden_states_values[:, -block_size:]
            
            hidden_states_keys[:, 1:] = self.hidden_state_key(prev_hidden_states[:, :-1])
            prev_hidden_states_keys = hidden_states_keys[:, -block_size:]

            pseudo_attention_scores = prev_hidden_states_keys * tok_emb
            embeddings_enrichment = pseudo_attention_scores * prev_hidden_state_values
            
            print(f"tok_emb_norm: {tok_emb.norm()}, enrich_norm: {embeddings_enrichment.norm()}, pos_norm: {pos_emb.norm()}")
            tok_emb = tok_emb + embeddings_enrichment
        
        
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
        
        return logits, loss, x
    
    def compile(self):
        if hasattr(torch, 'compile'):
            return torch.compile(self)
        return self
    
    def freeze_except(self, to_not_freeze):
        for name, param in self.named_parameters():
          if to_not_freeze in name:
            param.requires_grad = True
          else:
            param.requires_grad = False

# ----------------------------
# Load the pre-trained base model and compile
original_state_dict = torch.load('stateful_base_model.pt', map_location=torch.device(device))
filtered_state_dict = {k.replace('_orig_mod.', ''): v for k, v in original_state_dict.items()}

model = GPTLanguageModel()
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
model.load_state_dict(filtered_state_dict, strict=False)
model.to(device)
# model.freeze_except('hidden_state')
model.compile()

model_name = 'stateful_base_model.pt' if RECURSION_STEPS == 0 else 'stateful_model.pt'

optimizer_base = torch.optim.AdamW(model.parameters(), lr=learning_rate)

optimizers = []

for i in range(RECURSION_STEPS):
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizers.append(opt)

# ----------------------------
# Utility function to compute loss over an entire DataLoader
def evaluate_loss(model, dataloader, recursion_steps):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            logits, loss, hidden_states = model(xb, yb)
            for i in range(recursion_steps):
                logits, loss, hidden_states = model(xb, yb, hidden_states, i)
            total_loss += loss.item() * xb.size(0)
            count += xb.size(0)
    model.train()
    return total_loss / count

# ----------------------------
# Training loop (epoch-based)
best_val_loss = evaluate_loss(model, val_loader, RECURSION_STEPS)
print(f"initial_val_loss: {best_val_loss}")
for epoch in range(1, max_epochs + 1):
    model.train()
    running_loss = 0.0
    count = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, loss, hidden_states = model(xb, yb)
        optimizer_base.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_base.step()

        for i in range(RECURSION_STEPS):
            logits, loss, hidden_states = model(xb, yb, hidden_states.detach(), i)
            optimizers[i].zero_grad(set_to_none=True)
            loss.backward()
            optimizers[i].step()

        running_loss += loss.item() * xb.size(0)
        count += xb.size(0)

    train_loss = running_loss / count
    val_loss = evaluate_loss(model, val_loader, RECURSION_STEPS)

    print(f"Epoch {epoch}: Train loss {train_loss:.4f}, Val loss {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_name)

torch.save(model.state_dict(), model_name)

# ----------------------------
# Function to compute condition numbers for all 2D matrices in the model
def compute_condition_numbers(model):
    results = {}
    for name, param in model.named_parameters():
        if len(param.shape) == 2:
            try:
                matrix = param.detach().cpu()
                singular_values = torch.linalg.svdvals(matrix)
                max_sv = singular_values[0]
                min_sv = singular_values[-1]
                condition_num = float('inf') if min_sv == 0 else (max_sv / min_sv).item()
                results[name] = condition_num
            except Exception as e:
                results[name] = f"Error: {str(e)}"
    return results

# Example usage:
condition_numbers = compute_condition_numbers(model)
for name, cond in condition_numbers.items():
    print(f"{name}: {cond}")
