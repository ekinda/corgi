import torch
import torch.nn as nn

from .modules import FiLM, FiLM_MLP, TargetLengthCrop, ConvBlock, TransformerBlock

class Corgi(nn.Module):
    def __init__(self, config):
        super(Corgi, self).__init__()

        """
        Needed parameters in config:

        film_dimensions_conv
        film_dimensions_transformer
        film_mlp_hidden_layers
        film_mlp_dropout
        input_trans_regulators
        output_channels
        dim
        output_central_bins
        heads
        attn_dropout
        dropout_rate
        gqa (grouped query attention, number of key and value matrices per query matrix: int=2)
        loss_style (poisson_mn, poisson_nll, adaptive_loss)
        """
        self.config = config
        self.film_mlp_conv = FiLM_MLP(
            film_dim=config['film_dimensions_conv'],
            input_dim=config['input_trans_regulators'],
            hidden_dims=config['film_mlp_hidden_layers'],
            dropout=config['film_mlp_dropout']
        )
        self.film_mlp_transformer = FiLM_MLP(
            film_dim=config['film_dimensions_transformer'],
            input_dim=config['input_trans_regulators'],
            hidden_dims=config['film_mlp_hidden_layers'],
            dropout=config['film_mlp_dropout']
        )
        self.film_layers = nn.ModuleList()
        self.all_film_dimensions = config['film_dimensions_conv'] + config['film_dimensions_transformer']
        for cdim in self.all_film_dimensions:
            self.film_layers.append(FiLM(cdim))

        self.max_pool = nn.MaxPool1d(kernel_size = 2, padding = 0)
        self.conv_0 = nn.Conv1d(in_channels = 4,    out_channels = 896,  kernel_size = 15, padding="same")
        self.conv_1 = ConvBlock(in_channels = 896,  out_channels = 1024, kernel_size = 5)
        self.conv_2 = ConvBlock(in_channels = 1024, out_channels = 1152, kernel_size = 5)
        self.conv_3 = ConvBlock(in_channels = 1152, out_channels = 1280, kernel_size = 5)
        self.conv_4 = ConvBlock(in_channels = 1280, out_channels = 1440, kernel_size = 5)
        self.conv_5 = ConvBlock(in_channels = 1440, out_channels = 1536, kernel_size = 5)

        self.transformer_layers = nn.ModuleList([TransformerBlock(config) for _ in range(9)])

        self.crop = TargetLengthCrop(config['output_central_bins'])
        self.final_conv = nn.Sequential(
            ConvBlock(in_channels = config['dim'], out_channels = 1920, kernel_size = 1),
            nn.Dropout(0.1),
            nn.GELU(approximate='tanh'),
        )
        self.output_head = nn.Conv1d(in_channels = 1920, out_channels = config['output_channels'], kernel_size = 1)
        self.softplus = nn.Softplus()

        if config['loss_style'] in ['adaptive_loss', 'adaptive_mn']:
            self.loss_channel_weights = torch.nn.Parameter(torch.ones(config['output_channels']))

    def _transformer_film_wrapper(self, x, transformer_layer, film_layer, film_scales, film_shifts):
        x_out = transformer_layer.block[0](x)   # Attention
        orig_ff = transformer_layer.block[1].fn  
        x = orig_ff[0](x_out)  # LayerNorm
        x = x.permute(0, 2, 1)
        x = film_layer(x, film_scales, film_shifts)
        x = x.permute(0, 2, 1)
        x = orig_ff[1](x)  
        x = orig_ff[2](x)  
        x = orig_ff[3](x)  
        x = orig_ff[4](x)  
        x = orig_ff[5](x)  
        x = x + x_out
        return x
    
    def _split_film_params(self, film_params, film_dimensions):
        offset = 0
        scales = {}
        shifts = {}
        for i, cdim in enumerate(film_dimensions):
            scale = film_params[:, offset : offset+cdim]
            shift = film_params[:, offset+cdim : offset+2*cdim]
            offset += 2*cdim
            scales[i] = scale
            shifts[i] = shift
        return scales, shifts
    
    def convolutions(self, x, film_scales, film_shifts):
        x = self.conv_0(x)
        x = self.max_pool(x)
        x = self.film_layers[0](x, film_scales[0], film_shifts[0])
        x = self.conv_1(x)
        x = self.max_pool(x)
        x = self.conv_2(x)
        x = self.max_pool(x)
        x = self.film_layers[1](x, film_scales[1], film_shifts[1])
        x = self.conv_3(x)
        x = self.max_pool(x)
        x = self.conv_4(x)
        x = self.max_pool(x)
        x = self.conv_5(x)
        x = self.max_pool(x)
        return x
    
    def transformers(self, x, film_scales, film_shifts):
        x = self.transformer_layers[0](x)
        x = self.transformer_layers[1](x)
        x = self._transformer_film_wrapper(x, self.transformer_layers[2], self.film_layers[2], film_scales[0], film_shifts[0])
        x = self.transformer_layers[3](x)
        x = self.transformer_layers[4](x)
        x = self._transformer_film_wrapper(x, self.transformer_layers[5], self.film_layers[3], film_scales[1], film_shifts[1])
        x = self.transformer_layers[6](x)
        x = self.transformer_layers[7](x)
        x = self.transformer_layers[8](x)
        return x
    
    def forward(self, x, trans_reg):
        film_params_conv = self.film_mlp_conv(trans_reg)
        film_scales_conv, film_shifts_conv = self._split_film_params(film_params_conv, self.config['film_dimensions_conv'])
        film_params_tr = self.film_mlp_transformer(trans_reg)
        film_scales_tr, film_shifts_tr = self._split_film_params(film_params_tr, self.config['film_dimensions_transformer'])

        x = x.permute(0,2,1)                                                # (batch, 4, 524288)
        x = self.convolutions(x, film_scales_conv, film_shifts_conv)
        x = x.permute(0,2,1)                                                # (batch, 8192, 1536)
        x = self.transformers(x, film_scales_tr, film_shifts_tr)
        x = x.permute(0,2,1)                                                # (batch, 1536, 8192)
        x = self.crop(x)    
        x = self.final_conv(x)
        x = self.output_head(x)
        x = self.softplus(x)
        return x                                                            # (batch, output_channels [22], target_length [6144])