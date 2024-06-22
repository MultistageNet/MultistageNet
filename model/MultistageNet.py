import torch
import torch.nn as nn

from model.layers.depthwise_causalconv import Depthwise_CausalConvolution, Depthwise_CausalConvolution_regression
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
    def __init__(self, config, stage_cols):
        super(MultistageNet, self).__init__()

        self.n_stage = config.n_stage

        self.config = config

        d_model = config.d_model
        d_ff = config.d_ff
        kernel_size = config.kernel_size
        num_causal_layers = config.num_causal_layers
        activation = config.activation
        self.num_MMP_layers = config.num_mmp_layers

        seq_len = config.seq_len
        dropout_p = config.dropout_p
        pred_len = config.pred_len
        n_regressor_layer = config.n_regressor_layer
        num_heads = config.num_heads
        device = torch.device('cuda:'+ config.device_num)

        # define each stage dimension
        self.stage_w = [len(stage_cols[i]) for i in stage_cols.keys() if i != 'target']

        # define each stage DepthwiseCausalConv
        self.causalconvs = nn.ModuleList([Depthwise_CausalConvolution(self.stage_w[i], kernel_size, num_causal_layers, activation) for i in range(self.n_stage)])

        self.PE = PositionalEncoding(d_model, dropout_p)

        self.linear_embed = nn.ModuleList([nn.Linear(i, d_model) for i in self.stage_w])

        self.mmplayers = nn.ModuleList([MultistageLayer(d_model, num_heads, seq_len, d_ff, activation, self.n_stage, dropout_p, device) for _ in range(self.num_MMP_layers)])

        self.regressor_causalconv = Depthwise_CausalConvolution_regression(d_model, kernel_size, n_regressor_layer, activation, d_ff)

        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            getattr(nn, activation)(),
            nn.Linear(d_model//2, d_model//2),
            getattr(nn, activation)(),
            nn.Linear(d_model//2, pred_len))


    def forward(self, input_x):

        # Each stage input
        stage_inputs = []
        start, end = 0, self.stage_w[0]
        for i in range(1, self.n_stage):
            stage_inputs.append(input_x[:,:,start:end])
            start,end = end, end + self.stage_w[i]
        stage_inputs.append(input_x[:,:,start:end])

        caucalconv_outs = []
        for stage_num in range(self.n_stage):
            caucalconv_outs.append(self.causalconvs[stage_num](stage_inputs[stage_num]))

        linear_embeds = []
        for i in range(self.n_stage):
            tmp = caucalconv_outs[i]
            tmp = self.linear_embed[i](tmp)
            tmp = self.PE(tmp)
            linear_embeds.append(tmp)

        for j in range(self.num_MMP_layers):
            if j == 0:
                out_list = self.mmplayers[j](linear_embeds)
            else:
                out_list = self.mmplayers[j](out_list)

        output = self.regressor_causalconv(out_list[-1])
        output = self.regressor(output[:,-1,:])

        return output