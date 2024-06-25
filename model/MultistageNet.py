import torch
import torch.nn as nn

from model.layers.DepthwiseDilatedConv import DepthwiseDilatedConvolution, RegressorDepthwiseDilatedConvolution
from model.layers.PositionalEncoding import PositionalEncoding
from model.layers.Attention import InterstageAttention
from model.layers.GatedMechanism import GatedMechanism

class InterstageBlock(nn.Module):
    def __init__(self, d_model, num_heads, seq_len, d_ff, activation, dropout_p, device):
        super(InterstageBlock, self).__init__()

        self.attention = InterstageAttention(d_model, num_heads, seq_len, dropout_p, device)
        self.stagemix = GatedMechanism(d_model, d_ff, activation, dropout_p)

        self.layer_norm_prev = nn.LayerNorm(d_model)
        self.layer_norm_curr = nn.LayerNorm(d_model)

    def forward(self, x_prev, x_curr):

        att_out, _ = self.attention(x_prev, x_curr)

        x_prev = self.layer_norm_prev(x_prev + att_out)

        mix_out = self.stagemix(x_prev, x_curr)

        x_curr = self.layer_norm_curr(x_curr + mix_out)

        return (x_prev, x_curr)
    
class MultistageLayer(nn.Module):
    def __init__(self, d_model, num_heads, seq_len, d_ff, activation, n_stage, dropout_p, device):
        super(MultistageLayer, self).__init__()

        self.n_stage = n_stage

        self.stagemix = nn.ModuleList([InterstageBlock(d_model, num_heads, seq_len, d_ff, activation, dropout_p, device) for _ in range(1, self.n_stage)])


    def forward(self, x_list):

        out_list = []
        for i in range(0, self.n_stage-1):
            if i == 0:
                out = self.stagemix[i](x_list[i], x_list[i+1])
            else:
                out = self.stagemix[i](out[1], x_list[i+1])
            out_list.append(out[0])
        out_list.append(out[1])

        return out_list
    
class MultistageNet(nn.Module):
    def __init__(self, config, stage_vars):
        super(MultistageNet, self).__init__()

        self.config = config

        self.n_stage = config.n_stage
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.kernel_size = config.kernel_size
        self.activation = config.activation
        self.n_attention_heads = config.n_attention_heads
        self.n_temporal_layers = config.n_temporal_layers
        self.n_multistage_layers = config.n_multistage_layers
        self.n_regressor_layers = config.n_regressor_layers

        self.seq_len = config.seq_len
        self.dropout_p = config.dropout_p
        self.pred_len = config.pred_len
        
        device = torch.device('cuda:'+ config.device_num)

        # define each stage dimension
        self.stage_w = [len(stage_vars[i]) for i in stage_vars.keys() if i != 'target']

        # define each stage DepthwiseCausalConv
        self.temporal_layers = nn.ModuleList([DepthwiseDilatedConvolution(self.stage_w[i], self.kernel_size, self.n_temporal_layers, self.activation) for i in range(self.n_stage)])

        # define positional encoding
        self.PE = PositionalEncoding(self.d_model, self.dropout_p)

        # define linear embedding
        self.linear_embed = nn.ModuleList([nn.Linear(i, self.d_model) for i in self.stage_w])

        # define multistage layer
        self.multistage_layers = nn.ModuleList([MultistageLayer(self.d_model, self.n_attention_heads, self.seq_len, self.d_ff, self.activation, self.n_stage, self.dropout_p, device) for _ in range(self.n_multistage_layers)])

        # define regressor layer
        self.regressor_layers = RegressorDepthwiseDilatedConvolution(self.d_model, self.kernel_size, self.n_regressor_layers, self.activation, self.d_ff, self.pred_len)

    def forward(self, input_x):

        # Each stage input
        stage_inputs = []
        start, end = 0, self.stage_w[0]
        for i in range(1, self.n_stage):
            stage_inputs.append(input_x[:,:,start:end])
            start,end = end, end + self.stage_w[i]
        stage_inputs.append(input_x[:,:,start:end])

        # Temporal encoder
        temporal_outs = []
        for stage_num in range(self.n_stage):
            temporal_outs.append(self.temporal_layers[stage_num](stage_inputs[stage_num]))

        # Linear embedding
        linear_embeds = []
        for i in range(self.n_stage):
            tmp = temporal_outs[i]
            tmp = self.linear_embed[i](tmp)
            tmp = self.PE(tmp)
            linear_embeds.append(tmp)

        # Multistage layer
        for j in range(self.n_multistage_layers):
            if j == 0:
                out_list = self.multistage_layers[j](linear_embeds)
            else:
                out_list = self.multistage_layers[j](out_list)

        # Regressor
        output = self.regressor_layers(out_list[-1])

        return output