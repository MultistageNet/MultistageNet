import torch
import torch.nn as nn

class GatedMechanism(nn.Module):
    def __init__(self, d_model, d_ff, activation, dropout_p):
        super(GatedMechanism, self).__init__()

        self.prev_w = nn.Linear(d_model, d_ff)
        self.curr_w = nn.Linear(d_model, d_ff)

        self.activation = getattr(nn, activation)()

        self.fc_layer_1 = nn.Linear(d_ff, d_ff)

        self.fc_layer_2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x_prev, x_curr):
        x_prev = self.prev_w(x_prev)
        x_curr = self.curr_w(x_curr)

        g = torch.sigmoid(x_prev + x_curr)

        h = (1-g) * x_prev + g * x_curr

        x = self.fc_layer_1(self.dropout(h))

        x = self.activation(x)

        x = self.fc_layer_2(x)

        return x
