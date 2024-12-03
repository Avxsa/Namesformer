import torch
import torch.nn as nn

class MinimalTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, forward_expansion):
        super(MinimalTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, embed_size))
        num_layers = 3
        self.m_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads),
            num_layers=num_layers
        )
        self.w_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def forward(self, x, gender='m'):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.embed(x) + self.positional_encoding[:, :x.size(1), :]

        if gender == 'm':
            x = self.m_encoder(x)
        else:
            x = self.w_encoder(x)

        x = self.output_layer(x)
        return x