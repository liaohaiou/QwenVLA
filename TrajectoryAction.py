import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Scheduler import Scheduler
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim),
        )

    def forward(self, t: torch.Tensor):
        # t: [B] int64
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)  # [B, dim]

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        numerator = torch.exp(-torch.log(torch.tensor(10000))*(torch.arange(0, d_model, 2)/d_model))
        denominator = torch.arange(0, max_len).float().unsqueeze(1) * numerator
        pe[:, 0::2] = torch.sin(numerator * denominator)
        pe[:, 1::2] = torch.cos(numerator * denominator)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x : torch.Tensor):
        T = x.size(1)
        return self.pe[:, :T, :].to(x.dtype)


class CrossAttentionBlock(nn.Module):
    def __init__(self, num_heads, hidden_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, traj_embedding, condition_embedding):
        traj_embedding_norm = self.ln1(traj_embedding)
        traj_embedding_self_attn = self.self_attn(traj_embedding_norm, traj_embedding_norm, traj_embedding_norm)
        traj_embedding_self_attn = traj_embedding + traj_embedding_self_attn
        traj_embedding_self_attn_norm = self.ln2(traj_embedding_self_attn)
        traj_embedding_cross_attn = self.cross_attn(traj_embedding_self_attn_norm, condition_embedding, condition_embedding)
        traj_embedding_cross_attn = traj_embedding_cross_attn + traj_embedding_self_attn
        traj_embedding_cross_attn_norm = self.ln3(traj_embedding_cross_attn)
        traj_embedding_ffn = self.ffn(traj_embedding_cross_attn_norm)
        traj_embedding_ffn = traj_embedding_ffn + traj_embedding_cross_attn
        return traj_embedding_ffn

class ActionHead(nn.Module):
    def __init__(self, cfg : dict):
        super().__init__()
        self.traj_cordinate_dim = cfg['traj_cordinate_dim']
        self.d_model = cfg['d_model']
        self.num_heads = cfg['num_heads']
        self.num_layers = cfg['num_layers']
        self.traj_embdding_layer = nn.Linear(self.traj_cordinate_dim, self.d_model)
        self.time_emdding_layer = SinusoidalTimeEmbedding(self.d_model)
        self.position_embedding_layer = SinusoidalPositionEmbedding(24, self.d_model)
        self.cross_attn_layers = nn.ModuleList(
            [
                CrossAttentionBlock(self.num_heads, self.d_model)
                for _ in range(self.num_layers)
            ]
        )

        self.out = nn.Linear(self.d_model, self.traj_cordinate_dim)

    def forward(self, traj, timestep, condition_embedding):
        traj_embedding = self.traj_embdding_layer(traj)
        steps = traj_embedding.size(1)
        pe = self.position_embedding_layer(steps)
        traj_embedding = traj_embedding + pe
        te = self.time_emdding_layer(timestep)
        traj_embedding = traj_embedding + te

        for blk in self.cross_attn_layers:
            traj_embedding = blk(traj_embedding, condition_embedding)

        eps = self.out(traj_embedding)
        return eps



