import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanarFlow(nn.Module):
    def __init__(self, dim):
        """Instantiates one step of planar flow.

        Reference:
        Variational Inference with Normalizing Flows
        Danilo Jimenez Rezende, Shakir Mohamed
        (https://arxiv.org/abs/1505.05770)

        Args:
            dim: input dimensionality.
        """
        super(PlanarFlow, self).__init__()

        self.u = nn.Parameter(torch.randn(1, dim))
        self.w = nn.Parameter(torch.randn(1, dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """

        def m(x):
            return F.softplus(x) - 1.0

        def h(x):
            return torch.tanh(x)

        def h_prime(x):
            return 1.0 - h(x) ** 2

        inner = (self.w * self.u).sum()
        u = self.u + (m(inner) - inner) * self.w / self.w.norm() ** 2
        activation = (self.w * x).sum(dim=1, keepdim=True) + self.b
        x = x + u * h(activation)
        psi = h_prime(activation) * self.w
        log_det = torch.log(torch.abs(1.0 + (u * psi).sum(dim=1, keepdim=True)))

        return x, log_det


class RadialFlow(nn.Module):
    def __init__(self, dim):
        """Instantiates one step of radial flow.

        Reference:
        Variational Inference with Normalizing Flows
        Danilo Jimenez Rezende, Shakir Mohamed
        (https://arxiv.org/abs/1505.05770)

        Args:
            dim: input dimensionality.
        """
        super(RadialFlow, self).__init__()

        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1, dim))
        self.d = dim

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """

        def m(x):
            return F.softplus(x)

        def h(r):
            return 1.0 / (a + r)

        def h_prime(r):
            return -h(r) ** 2

        a = torch.exp(self.a)
        b = -a + m(self.b)
        r = (x - self.c).norm(dim=1, keepdim=True)
        tmp = b * h(r)
        x = x + tmp * (x - self.c)
        log_det = (self.d - 1) * torch.log(1.0 + tmp) + torch.log(
            1.0 + tmp + b * h_prime(r) * r
        )

        return x, log_det


class HouseholderFlow(nn.Module):
    def __init__(self, dim):
        """Instantiates one step of householder flow.

        Reference:
        Improving Variational Auto-Encoders using Householder Flow
        Jakub M. Tomczak, Max Welling
        (https://arxiv.org/abs/1611.09630)

        Args:
            dim: input dimensionality.
        """
        super(HouseholderFlow, self).__init__()

        self.v = nn.Parameter(torch.randn(1, dim))
        self.d = dim

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        outer = self.v.t() * self.v
        v_sqr = self.v.norm() ** 2
        H = torch.eye(self.d).cuda() - 2.0 * outer / v_sqr
        x = torch.mm(H, x.t()).t()

        return x, 0


class NiceFlow(nn.Module):
    def __init__(self, dim, mask, final=False):
        """Instantiates one step of NICE flow.

        Reference:
        NICE: Non-linear Independent Components Estimation
        Laurent Dinh, David Krueger, Yoshua Bengio
        (https://arxiv.org/abs/1410.8516)

        Args:
            dim: input dimensionality.
            mask: mask that determines active variables.
            final: True if the final step, False otherwise.
        """
        super(NiceFlow, self).__init__()

        self.final = final
        if final:
            self.scale = nn.Parameter(torch.zeros(1, dim))
        else:
            self.mask = mask
            self.coupling = nn.Sequential(
                nn.Linear(dim // 2, dim * 5),
                nn.ReLU(),
                nn.Linear(dim * 5, dim * 5),
                nn.ReLU(),
                nn.Linear(dim * 5, dim // 2),
            )

    def forward(self, x):
        if self.final:
            x = x * torch.exp(self.scale)
            log_det = torch.sum(self.scale)

            return x, log_det
        else:
            [B, W] = list(x.size())
            x = x.reshape(B, W // 2, 2)

            if self.mask:
                on, off = x[:, :, 0], x[:, :, 1]
            else:
                off, on = x[:, :, 0], x[:, :, 1]

            on = on + self.coupling(off)

            if self.mask:
                x = torch.stack((on, off), dim=2)
            else:
                x = torch.stack((off, on), dim=2)

            return x.reshape(B, W), 0


class Flow(nn.Module):
    def __init__(self, dim, type, length):
        """Instantiates a chain of flows.

        Args:
            dim: input dimensionality.
            type: type of flow.
            length: length of flow.
        """
        super(Flow, self).__init__()

        if type == "planar":
            self.flow = nn.ModuleList([PlanarFlow(dim) for _ in range(length)])
        elif type == "radial":
            self.flow = nn.ModuleList([RadialFlow(dim) for _ in range(length)])
        elif type == "householder":
            self.flow = nn.ModuleList([HouseholderFlow(dim) for _ in range(length)])
        elif type == "nice":
            self.flow = nn.ModuleList(
                [NiceFlow(dim, i // 2, i == (length - 1)) for i in range(length)]
            )
        else:
            self.flow = nn.ModuleList([])

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        [B, _] = list(x.size())
        log_det = torch.zeros(B, 1).cuda()
        for i in range(len(self.flow)):
            x, inc = self.flow[i](x)
            log_det = log_det + inc

        return x, log_det
