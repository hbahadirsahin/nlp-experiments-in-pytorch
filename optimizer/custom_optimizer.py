import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer


def warmup_cosine(x, warmup=0.002):
    s = 0
    if x < warmup:
        s = 1
    return s * (x / warmup) + (1 - s) * (0.5 * (1 + torch.cos(math.pi * x)))


def warmup_constant(x, warmup=0.002):
    s = 0
    if x < warmup:
        s = 1
    return s * (x / warmup) + (1 - s)


def warmup_linear(x, warmup=0.002):
    s = 0
    if x < warmup:
        s = 1
    return (s * (x / warmup) + (1 - s)) * (1 - x)


SCHEDULES = {
    "cos": warmup_cosine,
    "constant": warmup_constant,
    "linear": warmup_linear,
}


class NoamOptimizer():
    # Direct c/p from Attention is All You Need notebook (famous Harvard's one)
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        # Update parameters and rate
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


class OpenAIAdam(Optimizer):
    # Referance to https://github.com/huggingface
    # Open AI version of Adam with weight decay
    def __init__(self, params, lr, schedule, warmup, t_total, b1=0.9, b2=0.999, e=1e-8, l2=0, vector_l2=False,
                 max_grad_norm=-1, **kwargs):
        assert 0 < lr
        assert schedule == "cos" or schedule == "constant" or schedule == "linear"
        assert 0 < warmup
        assert 0 < b1 <= 1.0
        assert 0 < b2 <= 1.0
        assert 0 < e

        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total, b1=b1, b2=b2, e=e, l2=l2,
                        vector_l2=vector_l2, max_grad_norm=max_grad_norm)

        super(OpenAIAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        # Performs a single optimization step
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.date
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients. Use SparseAdam")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["b1"], group["b2"]

                state["step"] += 1

                # Add grad clipping
                if group["max_grad_norm"] > 0:
                    clip_grad_norm_(p, group["max_grad_norm"])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group["e"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                schedule_fn = SCHEDULES(group["schedule"])
                lr_scheduled = group["lr"] * schedule_fn(state["state"] / group["t_total"], group["warmup"])
                step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Add weight decay at the end (fixed version)
                if (len(p.size()) > 1 or group["vector_l2"]) and group["l2"] > 0:
                    p.data.add_(-lr_scheduled * group["l2"], p.data)
        return loss


class Padam(Optimizer):
    """Partially Adaptive Momentum Estimation algorithm"""

    def __init__(self, params, lr, amsgrad, e=1e-8, b1=0.9, b2=0.999, partial=0.25, weight_decay=0, max_grad_norm=-1,
                 **kwargs):
        assert 0 < lr
        assert 0 < b1 <= 1.0
        assert 0 < b2 <= 1.0
        assert 0 < e
        assert 0 < partial <= 0.5
        defaults = dict(lr=lr, b1=b1, b2=b2, e=e, amsgrad=amsgrad, partial=partial, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(Padam, self).__init__(params, defaults)

    def step(self, closure=None):
        # Performs a single optimization step
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients. Use SparseAdam")

                amsgrad = group['amsgrad']
                partial = group['partial']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["b1"], group["b2"]
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                state["step"] += 1

                # Add grad clipping
                if group["max_grad_norm"] > 0:
                    clip_grad_norm_(p, group["max_grad_norm"])

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['e'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["e"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom ** (partial * 2))
        return loss


if __name__ == '__main__':
    opts = [NoamOptimizer(512, 1, 4000, None),
            NoamOptimizer(512, 1, 8000, None),
            NoamOptimizer(256, 1, 4000, None)]
    plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    plt.legend(["512:4000", "512:8000", "256:4000"])
    plt.show()
