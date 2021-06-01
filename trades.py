import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):

    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)
    norm = distance.split('_')[1]

    # random start
    delta = torch.rand_like(x_natural) * 2 * epsilon - epsilon
    if distance != 'l_inf':  # projected into feasible set if needed
        normVal = torch.norm(delta.view(batch_size, -1), int(norm), 1)
        normVal = torch.max(normVal, torch.ones_like(norm) * 1e-6)
        mask = normVal <= epsilon
        scaling = epsilon / normVal
        scaling[mask] = 1
        delta = delta * scaling.view(batch_size, 1, 1, 1)

    x_adv = x_natural.detach() + delta.detach()

    # PGD with KL loss
    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            _, logit_adv = model(x_adv)
            _, logit_natural = model(x_natural)
            loss_kl = criterion_kl(F.log_softmax(logit_adv, dim=1),
                                   F.softmax(logit_natural, dim=1))
        updates = torch.autograd.grad(loss_kl, [x_adv])[0]
        if distance == 'l_inf':
            updates = torch.sign(updates)
        else:
            normVal = torch.norm(updates.view(batch_size, -1), int(norm), 1)
            normVal = torch.max(normVal, torch.ones_like(norm) * 1e-6)
            updates = updates / normVal.view(batch_size, 1, 1, 1)

        updates = updates * step_size
        x_adv = x_adv.detach() + updates.detach()

        # projection
        delta = x_adv - x_natural
        if distance == 'l_inf':
            delta = torch.clamp(delta, -epsilon, epsilon)
        else:
            normVal = torch.norm(delta.view(batch_size, -1), int(norm), 1)
            normVal = torch.max(normVal, torch.ones_like(norm) * 1e-6)
            mask = normVal <= epsilon
            scaling = epsilon / normVal
            scaling[mask] = 1
            delta = delta * scaling.view(batch_size, 1, 1, 1)

        x_adv = torch.clamp(x_natural+delta, 0.0, 1.0)

    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    _, logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    _, logit_adv = model(x_adv)
    _, logit_natural = model(x_natural)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logit_adv, dim=1),
                                                    F.softmax(logit_natural, dim=1))
    loss = loss_natural + beta * loss_robust
    return loss, loss_natural, loss_robust
