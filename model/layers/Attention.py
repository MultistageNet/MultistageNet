import torch
import torch.nn as nn

class InterstageAttention(nn.Module):
    def __init__(self, d_model, num_heads, seq_len, dropout_p, device):
        super(InterstageAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        self.mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)

        self.dropout = nn.Dropout(dropout_p)

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)

    def forward(self, x_prev, x_curr):

        q = self.fc_q(x_curr)
        k = self.fc_k(x_prev)
        v = self.fc_v(x_prev)

        q = q.view(q.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.view(k.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.view(v.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
            
        scores = torch.matmul(q, k.transpose(-2,-1)) / self.scale

        masked_scores = scores.masked_fill(self.mask==1, -1e9)

        att_score = torch.softmax(masked_scores, dim=-1)

        x = torch.matmul(self.dropout(att_score), v)
        
        x = x.transpose(1, 2).contiguous()

        x = x.view(x.size(0), -1, self.d_model)

        return x, att_score