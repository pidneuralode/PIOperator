# Here we will set some physics loss functions for PDE equations given in the experiments
# Usually there will be IC loss, BC loss and Residual loss, combined with PDE physical information
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import utils.gradient as grad


class Burgers2dPhyLoss:
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.loss_function = nn.MSELoss(reduction='mean')

    def __call__(self, ics_batch, bcs_batch, res_batch):
        """
        compute the loss of ics, bcs and res
        :param ics_batch:
        :param bcs_batch:
        :param res_batch:
        :return:
        """
        loss_ics = self.loss_ics(ics_batch)
        loss_bcs = self.loss_bcs(bcs_batch)
        loss_res = self.loss_res(res_batch)
        # todo: there are some fixed weights that need to be adjusted
        loss = 20 * loss_ics + loss_bcs + loss_res

        return loss

    def compute_ds_dx(self, u, y: torch.Tensor):
        # Define ds/dx
        # y = (t, x)
        t, x = y[:, 0], y[:, 1]
        if t.requires_grad is False:
            t.requires_grad = True
        if x.requires_grad is False:
            x.requires_grad = True
        y = torch.stack((t, x), dim=1)
        s = self.model((u, y))
        return autograd.grad(s, x, grad_outputs=torch.ones_like(s), retain_graph=True, create_graph=True)[0]

    def compute_PDE_residual(self, u, y):
        # Define ds/dx
        # y = (t, x)
        t, x = y[:, 0], y[:, 1]
        t.requires_grad = True
        x.requires_grad = True
        y = torch.stack((t, x), dim=1)
        s = self.model((u, y))
        ds_dt = autograd.grad(s, t, grad_outputs=torch.ones_like(s), retain_graph=True, create_graph=True)[0]
        ds_dx = self.compute_ds_dx(u, y)
        ds_dxx = autograd.grad(ds_dx, x, grad_outputs=torch.ones_like(ds_dx), retain_graph=True, create_graph=True)[0]

        # here coff v for this burgers is 0.01
        res = ds_dt + s * ds_dx - 0.01 * ds_dxx

        return res

    def loss_ics(self, ics_batch):
        """
        define initial loss
        :param ics_batch: ((u,y), outputs)
        :return:
        """
        (u, y), outputs = ics_batch

        # compute forward pass
        s_pred = self.model((u, y))

        ics_loss = self.loss_function(s_pred.flatten(), outputs.flatten())

        return ics_loss

    def loss_bcs(self, bcs_batch):
        """
        define boundary loss
        :param bcs_batch:
        :return:
        """
        inputs, outputs = bcs_batch
        u, y = inputs

        # compute forward pass
        # s(0,t) = s(1,t)
        s_bc1_pred = self.model((u, y[:, :2]))
        s_bc2_pred = self.model((u, y[:, 2:]))

        # ds/dx(0,t) = ds/dx(1,t)
        ds_dx_bc1_pred = self.compute_ds_dx(u, y[:, :2])
        ds_dx_bc2_pred = self.compute_ds_dx(u, y[:, 2:])

        # compute boundary loss
        return self.loss_function(s_bc1_pred, s_bc2_pred) + self.loss_function(ds_dx_bc1_pred, ds_dx_bc2_pred)

    def loss_res(self, res_batch):
        """
        define residual loss
        :param res_batch:
        :return:
        """
        inputs, outputs = res_batch
        u, y = inputs

        # compute forward pass
        s_pred = self.compute_PDE_residual(u, y)

        # compute loss
        return self.loss_function(s_pred, outputs)

