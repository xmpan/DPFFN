import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(CrossAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, query, key):
        value = key
        batch_size = query.size(0)
        Q = self.W_Q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)

        # Concat heads
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)

        # Final linear layer
        output = self.fc(context)
        return output, attention