import torch
import torch.nn as nn
from flash_attn.modules.mha import MHA

class Residual(torch.nn.Module):
    """
    Residual connection around any layer.
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    
class FiLM(nn.Module):
    """
    FiLM layer. Modulates the input layer by scale and shift parameters.

    Args:
        num_channels (int): Number of channels of the FiLM layer.
            Must have the same dimensions as scale and shift when calling forward().
    """
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, x, scale, shift):
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        return x * scale + shift

class FiLM_MLP(nn.Module):
    """
    Multilayer perceptron to generate weights of all FiLM layers in the network.

    Args:
        config (dict): Dictionary containing configuration settings.
            Relevant are the dimensions of film layers, number of channels of hidden layers, input dim and dropout.
    """
    def __init__(self, film_dim, input_dim, hidden_dims, dropout):
        super().__init__()

        self.film_dimensions = film_dim
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.output_dim = 2 * sum(self.film_dimensions)

        layers = [
            nn.BatchNorm1d(input_dim)
        ]
        prev_dim = self.input_dim
        for hd in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.GELU(approximate='tanh'))
            layers.append(nn.Dropout(self.dropout))
            prev_dim = hd

        self.film_network = nn.Sequential(*layers)
        self.final_layer = nn.Linear(prev_dim, self.output_dim)

    def forward(self, x):
        x = self.film_network(x)
        x = self.final_layer(x)
        x = torch.tanh(x/3) * 3.0         # Limits output to range [-3,3]
        return x                          # (B, output_dim)

class FlashAttention(nn.Module):
    """
    Wrapper around MHA from the flash_attn package.
    """
    def __init__(
        self,
        dim = 1536,
        heads = 8,
        dropout = 0.15,
        rotary_emb_base = 20000.0,
        rotary_emb_scale_base = None,
        gqa = 1
        ):
        super().__init__()

        self.mha = MHA(
            use_flash_attn=True,
            embed_dim=dim,
            num_heads = heads,
            num_heads_kv = (heads//gqa),   # Grouped query attention with multiple Q per 1 K&V
            qkv_proj_bias=True,
            out_proj_bias=True,
            dropout=dropout,
            softmax_scale=(dim/heads) ** -0.5,
            causal=False,
            rotary_emb_dim=128,
            rotary_emb_base=rotary_emb_base,
            rotary_emb_scale_base = rotary_emb_scale_base,
            fused_bias_fc = False,
        ) 

        nn.init.xavier_normal_(self.mha.Wqkv.weight)
        nn.init.xavier_normal_(self.mha.out_proj.weight)
        nn.init.zeros_(self.mha.Wqkv.bias)
        nn.init.zeros_(self.mha.out_proj.bias)


    def forward(self, x):
        out = self.mha(x)
        return out
    
class TargetLengthCrop(nn.Module):
    """
    Crops the input tensor to the target length by removing equal amounts from both ends.
    Crops the sequence length dimension, which is assumed to be the last dimension of the input tensor.
    """
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-1], self.target_length
        if target_len == -1:
            return x
        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')
        crop_amount = (seq_len - target_len) // 2
        return x[..., crop_amount:crop_amount + target_len]
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=1):
        super(ConvBlock, self).__init__()
        self.norm = nn.BatchNorm1d(in_channels, eps = 0.001)
        self.activation = nn.GELU(approximate='tanh')
        self.conv_layer = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding='same')
            
    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv_layer(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.block = nn.Sequential(
            Residual(nn.Sequential(
                nn.LayerNorm(config['dim'], eps = 0.001),
                FlashAttention(
                    config['dim'],
                    heads = config['heads'],
                    dropout = config['attn_dropout'],
                    gqa = config['gqa']
                ),
                nn.Dropout(0.2))
            ),
            Residual(nn.Sequential(
                nn.LayerNorm(config['dim'], eps = 0.001),
                nn.Linear(config['dim'], config['dim'] * 2),
                nn.Dropout(config['dropout_rate']),
                nn.ReLU(),
                nn.Linear(config['dim'] * 2, config['dim']),
                nn.Dropout(config['dropout_rate'])
            )))

    def forward(self, x):
        return self.block(x)