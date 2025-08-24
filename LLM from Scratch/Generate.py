import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Device ----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# ---------------- Hyperparameters ----------------
block_size = 64
batch_size = 128
max_iters = 3000
learning_rate = 3e-4
eval_iters = 100
dropout = 0.2
n_layer = 8
n_embd = 384
n_head = 8

# ---------------- Transformer Components ----------------
class Head(nn.Module):
    """ One head of self-attention """
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
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ Simple FFN: Linear -> ReLU -> Linear """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: attention + feedforward """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.ln1(x + self.sa(x))
        x = self.ln2(x + self.ffwd(x))
        return x

#GPT Model ----------------
class GPTLanguageModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocabulary_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)

    def forward(self, index, target=None):
        B, T = index.shape

        # 🔹 Truncate if input is longer than block_size
        if T > block_size:
            index = index[:, -block_size:]
            T = block_size

        tok_emb = self.token_embedding_table(index)     # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if target is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = target.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, index, max_new_tokens, temperature=1.0, top_k=50):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(index)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            if top_k is not None:
                v, ix = torch.topk(probs, k=top_k)
                probs = torch.zeros_like(probs).scatter_(1, ix, v)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)

            # 🔹 Always keep only the last block_size tokens
            if index.size(1) > block_size:
                index = index[:, -block_size:]

        return index

#Load Vocabulary 
with open("vocab.pkl", "rb") as f:
    string_to_int, int_to_string = pickle.load(f)

vocabulary_size = len(string_to_int)
UNK_ID = string_to_int.get("<unk>", None)

# Load Trained Model 
model = GPTLanguageModel(vocabulary_size)
model.apply(model._init_weights)
model.load_state_dict(torch.load("wizard_gpt.pth", map_location=device))
model.to(device)
model.eval()

#  Helpers
def encode(s):
    ids = []
    for c in s:
        if c in string_to_int:
            ids.append(string_to_int[c])
        elif UNK_ID is not None:
            ids.append(UNK_ID)
    return ids

def decode(indices):
    return "".join([int_to_string[i] for i in indices if i in int_to_string])

# Prompt
prompt = "Dorothy stepped into the forest and"
prompt = prompt.lower()
context_ids = encode(prompt)

# Debug: check vocab range
bad_ids = [i for i in context_ids if i >= vocabulary_size]
if bad_ids:
    raise ValueError(f"Found token IDs out of range: {bad_ids}. Vocab mismatch!")

context = torch.tensor([context_ids], dtype=torch.long, device=device)

#Generate
output = model.generate(context, max_new_tokens=1000, temperature=0.8, top_k=50)[0].tolist()

#Print
print("\n Generated Text:\n")
text = decode(output)   # convert token IDs -> string
print(text)

with open("Output.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Generated text saved to Output.txt")