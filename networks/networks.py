from torch import nn
from torch.nn import init


class Dense(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 activation=nn.ELU(),
                 using_norm=True,
                 initialization=None
                 ):
        super().__init__()
        self.using_norm = using_norm
        self.init = initialization
        self.linear = nn.Linear(dim_in, dim_out)
        if self.using_norm:
            self.norm = nn.BatchNorm1d(dim_out)
        self.activation = activation
        self._init_weight()

    def _init_weight(self):
        if self.init == "uniform":
            init.uniform_(self.linear.weight, a=-1., b=1.)
        elif self.init == "normal":
            init.normal_(self.linear.weight, mean=0, std=1)
        elif self.init == "xavier":
            init.xavier_uniform_(self.linear.weight, gain=0.2)
        elif self.init == "orthogonal":
            init.orthogonal_(self.linear.weight, gain=0.2)
        elif self.init == "kaiming":
            init.kaiming_uniform_(self.linear.weight, mode='fan_in', nonlinearity='elu')
        else:
            pass
        init.zeros_(self.linear.bias)

    def forward(self, x):
        if self.using_norm:
            x = self.norm(self.linear(x))
        else:
            x = self.linear(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class MLP(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_hidden,
                 dim_out,
                 n_layers=2,
                 act=nn.ELU(),
                 output_act=None,
                 **kwargs):
        super().__init__()
        n_hidden = n_layers - 2
        input_layer = Dense(dim_in=dim_in, dim_out=dim_hidden, activation=act, **kwargs)
        layers = [input_layer]
        for i in range(n_hidden):
            layers.append(Dense(dim_in=dim_hidden, dim_out=dim_hidden, activation=act, **kwargs))
        output_layer = Dense(dim_in=dim_hidden, dim_out=dim_out, activation=output_act, **kwargs)
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)

        return x
