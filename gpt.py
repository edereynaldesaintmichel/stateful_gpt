import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 1024 # how many independent sequences will we process in parallel?
block_size = 33 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 5e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_embd = 64
n_head = 8
n_layer = 3
dropout = 0.2
contribution_coeff = 1
# ------------

# torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('combined_gutenberg_10M.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss, hidden_states = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

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
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        dim = int(4.2 * n_embd)
        self.net = nn.Sequential(
            nn.Linear(n_embd, dim),
            nn.ReLU(),
            nn.Linear(dim, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
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

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_s = nn.LayerNorm(n_embd) # token_embeddings layer norm
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.relu = nn.ReLU()

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, prev_hidden_states=None, contribution_coeff=1):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb:torch.Tensor = self.token_embedding_table(idx) # (B,T,C)
        # tok_emb = self.ln_s(tok_emb)

        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, x

    def generate(self, idx, max_new_tokens, use_recursion=False, contribution_coeff=1):
        # idx is (B, T) array of indices in the current context
        hidden_states = None
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                logits, loss, tmp_hidden_states = self.forward(idx_cond, None, hidden_states, contribution_coeff)
                if (hidden_states is None):
                  hidden_states = torch.zeros(tmp_hidden_states.size(0), 1, tmp_hidden_states.size(2), device=tmp_hidden_states.device, dtype=tmp_hidden_states.dtype)
                
                if use_recursion:
                    padding = torch.zeros(hidden_states.size(0), 1, hidden_states.size(2), device=hidden_states.device, dtype=hidden_states.dtype)
                    hidden_states = torch.cat((hidden_states[:,:-1], tmp_hidden_states[:,-1:], padding), dim=1)[:,-block_size:]
                    # hidden_states = torch.cat((tmp_hidden_states, padding), dim=1)[:,-block_size:,:]
                else:
                    hidden_states = None
                logits = logits[:, -1, :] * 2 # becomes (B, C) temperature is set to 1/2 = .5
                probs = F.softmax(logits, dim=-1) # (B, C)
                idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            return idx
        
    def freeze_except_logits_embedding(self):
        for name, param in self.named_parameters():
            if 'hidden_state_key_proj' not in name and 'hidden_state_value' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

# Load the original state dict
original_state_dict = torch.load('stateful_base_model.pt', map_location=torch.device(device))

# Load the filtered state dict
model = GPTLanguageModel()
model = model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
model.load_state_dict(original_state_dict, strict=False)

############################## Base model training

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    recursion_steps = 0 #int(iter / max_iters * RECURSION_STEPS)
    best_loss = float('inf')
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_loss:
            best_loss = losses['val']
            torch.save(model.state_dict(), 'stateful_base_model.pt')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss, hidden_states = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'stateful_base_model.pt')

model = GPTLanguageModel()
m = model.to(device)
model.load_state_dict(torch.load('stateful_base_model.pt'))
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
with open('scaled_gpt_more.txt', 'w+') as file:
    file.write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))