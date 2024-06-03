# Here we will set some physics loss functions for PDE equations given in the experiments
# Usually there will be IC loss, BC loss and Residual loss, combined with PDE physical information
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class Burgers1dPhyLoss(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, ics_batch, bcs_batch, res_batch):
        """
        compute the loss of ics, bcs and res
        :param ics_batch:
        :param bcs_batch:
        :param res_batch:
        :return:
        """
        loss_ics = self.loss_ics(ics_batch)
        loss_bcs = self.loss_bs(bcs_batch)
        loss_res = self.loss_res(res_batch)
        # todo: there are some fixed weights that need to be adjusted
        loss = 20 * loss_ics + loss_bcs + loss_res

        return loss

    def compute_ds_dx(self, u, y):
        # Define ds/dx
        # y = (t, x)
        s = self.model((u, y))
        return autograd.grad(s, y[:, 1], retain_graph=True, create_graph=True)

    def compute_PDE_residual(self, u, y):
        # Define ds/dx
        # y = (t, x)
        s = self.model((u, y))
        ds_dt = autograd.grad(s, y[:, 0], retain_graph=True, create_graph=True)
        ds_dx = self.compute_ds_dx(u, y)
        ds_dxx = autograd.grad(ds_dx, y[:, 1], retain_graph=True)

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

        ics_loss = F.mse_loss(s_pred.flatten() - s_pred)

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
        # s(x,0) = s(x,1)
        s_bc1_pred = self.model((u, y[0]))
        s_bc2_pred = self.model((u, y[1]))

        # ds/dx(0,t) = ds/dx(1,t)
        ds_dx_bc1_pred = self.compute_ds_dx(u, y[0])
        ds_dx_bc2_pred = self.compute_ds_dx(u, y[1])

        # compute boundary loss
        return F.mse_loss(s_bc1_pred, s_bc2_pred) + F.mse_loss(ds_dx_bc1_pred, ds_dx_bc2_pred)

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
        return F.mse_loss(s_pred, outputs)

