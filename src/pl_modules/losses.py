import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F



def nll_loss(yhat, y, x):
    """Wrapper for the normal nll loss."""
    return F.nll_loss(yhat, y)


def bce_loss(yhat, y, x):
    """Wrapper for the normal nll loss."""
    return nn.BCELoss()(yhat.float(), y.float())


def cce_loss(yhat, y, x):
    """Wrapper for the normal nll loss."""
    return nn.CrossEntropyLoss(reduction="none")(yhat, y)


def celltype_cce_loss(yhat, y, x, poly_channel=0, dapi_channel=1):
    """Wrapper nll loss with a mask on the dapi channel.

    Only compute loss at locations where you have DAPI + Label."""
    weight = [0.10908450970083867, 9.344753437619014, 0, 9.516677251173297, 9.800194298602854, 16.04023407840591, 18.326204034372797, 13.377497741641083, 25.548119675798734, 31.184488811544636, 3.9093832611950297]
    # weight = [0.2727112742520967, 33.443744354102705, 9.773458152987574, 4.977617082171507]
    # weight = [0.34472822905947637, 44.59165913880361, 13.031277537316766]
    weight = [0.34401008173185593, 61.08734678124265, 13.031277537316766]
    weight = torch.tensor(weight).float().to(yhat.device)
    dapi = x[:, dapi_channel]
    # poly = x[:, poly_channel]
    dapi_fg = dapi > 0.065  # 0.070
    dapi_bg = dapi < 0.019
    fg_mask = torch.logical_and(dapi_fg, y > 0)
    bg_mask = torch.logical_and(dapi_bg, y == 0)
    mask = torch.logical_or(fg_mask, bg_mask)
    # from matplotlib import pyplot as plt;plt.subplot(131);plt.imshow(dapi.squeeze().cpu().detach());plt.subplot(122);plt.imshow(mask.squeeze().detach().cpu());plt.show()
    loss = nn.CrossEntropyLoss(reduction="none")(yhat, y)
    # loss = nn.CrossEntropyLoss(reduction="none", weight=weight)(yhat, y)
    loss = loss * mask.float()
    return loss


def nt_xent_loss(out_1, out_2, temperature=0.1, eps=1e-6):
    """
    assume out_1 and out_2 are normalized
    out_1: [batch_size, dim]
    out_2: [batch_size, dim]
    """
    # gather representations in case of distributed training
    # out_1_dist: [batch_size * world_size, dim]
    # out_2_dist: [batch_size * world_size, dim]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        out_1_dist = SyncFunction.apply(out_1)
        out_2_dist = SyncFunction.apply(out_2)
    else:
        out_1_dist = out_1
        out_2_dist = out_2

    # out: [2 * batch_size, dim]
    # out_dist: [2 * batch_size * world_size, dim]
    out = torch.cat([out_1, out_2], dim=0)
    out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

    # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
    # neg: [2 * batch_size]
    cov = torch.mm(out, out_dist.t().contiguous())
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)

    # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
    row_sub = Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

    # Positive similarity, pos becomes [2 * batch_size]
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / (neg + eps)).mean()

    return loss


class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]

