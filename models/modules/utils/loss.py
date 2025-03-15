import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, Any, Tuple, List, Dict

from .macro import *
from .masking import _get_padding_mask, _get_visibility_mask

class CadLoss(nn.Module):
    def __init__(self, type="primitive"):
        super().__init__()
        assert type in ("primitive", "feature")
        self.type = type
        if type == "primitive":
            self.n_cmd = N_PRIM_COMMANDS + 2
            self.d_param = PARAM_DIM
            self.register_buffer("cmd_param_mask", torch.tensor(PRIM_PARAM_MASK))
        else:
            self.n_cmd = N_FEAT_COMMANDS + 2
            self.d_param = PARAM_DIM
            self.register_buffer("cmd_param_mask", torch.tensor(FEAT_PARAM_MASK))

    def forward(self, output, label_cmd, label_param, param_weight):
        # Target & predictions
        tgt_cmd, tgt_param = label_cmd, label_param
        if self.type == "primitive":
            cmd_logits, param_logits = output["prim_cmd"], output["prim_param"]
        else:
            cmd_logits, param_logits = output["feat_cmd"], output["feat_param"]

        visibility_mask = _get_visibility_mask(tgt_cmd, seq_dim=-1)
        padding_mask_ = _get_padding_mask(tgt_cmd, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)
        mask = self.cmd_param_mask[tgt_cmd.long()]

        loss_cmd = F.cross_entropy(
            cmd_logits[padding_mask_.bool()].reshape(-1, self.n_cmd),
            tgt_cmd[padding_mask_.bool()].reshape(-1).long(),
        )

        loss_param = F.cross_entropy(
            param_logits[mask.bool()].reshape(-1, self.d_param),
            tgt_param[mask.bool()].reshape(-1).long() + 1
        )

        return {"loss_cmd": loss_cmd, "loss_param": loss_param*param_weight}


class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        input1_ = torch.mean(input1, dim=0)
        input1 = input1 - input1_
        input1 = torch.nn.functional.normalize(input1, p=2, dim=1)

        input2_ = torch.mean(input2, dim=0)
        input2 = input2 - input2_
        input2 = torch.nn.functional.normalize(input2, p=2, dim=1)

        input1 = torch.transpose(input1, 0, 1)
        correlation_matrix = torch.matmul(input1, input2)

        diff_loss = torch.mean(torch.square(correlation_matrix)) * 1.0
        if diff_loss < 0.0:
            diff_loss = 0.0
        return diff_loss


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.0):
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    def __init__(
        self, 
        alpha: Optional[float] = 1.0, 
        lo: Optional[float] = 0.0, 
        hi: Optional[float] = 1.,
        max_iters: Optional[int] = 1000., 
        auto_step: Optional[bool] = False
    ):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor):
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1


class DomainAdversarialLoss(nn.Module):
    """
    Domain-Adversarial Training of Neural Networks (ICML 2015) 
    <https://arxiv.org/abs/1505.07818>
    """
    def __init__(
        self, 
        discriminator: nn.Module, 
        reduction: Optional[str] = 'mean',
        grl: Optional[nn.Module] = None
    ):
        super(DomainAdversarialLoss, self).__init__()
        self.discriminator = discriminator
        self.grl = grl
        if grl is None:
            self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.bce = lambda input, target, weight: F.binary_cross_entropy(input, target, weight, reduction=reduction)
    
    def forward(
        self, 
        f_s: torch.Tensor, 
        f_t: torch.Tensor,
        w_s: Optional[torch.Tensor] = None, 
        w_t: Optional[torch.Tensor] = None
    ):
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.discriminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)

        if w_s is None:
            w_s = torch.ones_like(d_label_s)
        if w_t is None:
            w_t = torch.ones_like(d_label_t)
        return 0.5 * (self.bce(d_s, d_label_s, w_s.view_as(d_s)) + self.bce(d_t, d_label_t, w_t.view_as(d_t)))


class DomainDiscriminator(nn.Sequential):
    """
    Domain-Adversarial Training of Neural Networks (ICML 2015) 
    <https://arxiv.org/abs/1505.07818>
    """
    def __init__(self, in_feature: int, hidden_size: int, batch_norm=True):
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )

    def get_parameters(self):
        return [{"params": self.parameters(), "lr": 1.}]