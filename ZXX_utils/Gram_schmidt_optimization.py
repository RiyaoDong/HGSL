import torch
import torch.nn as nn
from torch.autograd import Function

def DGLoss(W):
    W_1, W_2 = W.size()
    total = W_1**2 - W_1
    W_mm = torch.mm(W, W.t()) * (torch.ones(W_1) - torch.eye(W_1)).cuda()
    loss = torch.sum(torch.abs(W_mm).view(-1)) / total
    return loss

def proj(u, v):
    u = u.unsqueeze(1)
    v = v.unsqueeze(1)
    w = (v.t().mm(u) / u.t().mm(u)) * u
    w = w.squeeze(1)
    return w

class Gram_s_optimization(Function):
    @staticmethod
    def forward(ctx, input):
        x = input
        W_1, W_2 = x.size()
        total = W_1**2 - W_1
        W_mm = torch.mm(x, x.t()) * (torch.ones(W_1) - torch.eye(W_1)).cuda()
        y = torch.sum(torch.abs(W_mm).view(-1)) / total
        ctx.save_for_backward(input)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        W = input[0]
        W_1, W_2 = W.size()
        total = W_1 ** 2 - W_1
        for ii in range(W_1):
            W[ii, :] = W[ii, :] / torch.norm(W[ii, :])
            for jj in range(ii,W_1):
                if jj == ii:
                    continue
                W[jj, :] = W[jj, :] - proj(W[ii, :], W[jj, :])
        grad_input = -1 * grad_output * W / total
        #WW = input.size()
        return grad_input

def Gram_s_optim_Layer(var):
    return Gram_s_optimization.apply(var)
