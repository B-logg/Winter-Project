from cmath import nan
import torch
from torch import nn
import sys, os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from evaluate import corr, r_square, corr_wZero, r_square_wZero


class CarbonLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, cls_lambda=50, reg_lambda=0.005, ignore_label=255):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.cls_lambda = cls_lambda
        self.reg_lambda = reg_lambda

    def forward(self, input_cls,input_reg, target_cls, target_reg):
                
        cls_loss = self.ce(input_cls, target_cls.type(torch.long).squeeze())
        
        reg_loss = self.mse(
            torch.flatten(input_reg, end_dim=-2),
            torch.flatten(target_reg, end_dim=-2)
        )
        input_reg = input_reg.squeeze()
        target_reg = target_reg.squeeze()

        corr_sum=0
        r_sum =0
        idx = 0
        for i in range(len(input_reg)):
            ir = input_reg[i,...].cpu().detach().numpy()
            tr = target_reg[i,...].cpu().detach().numpy()
            if np.count_nonzero(tr) == 0:
                continue          
            corr_res = corr_wZero(ir,tr)
            r_res = r_square_wZero(ir,tr)
            if np.isnan(corr_res) or np.isnan(r_res):           
                continue
            corr_sum = corr_sum + corr_res
            r_sum = r_sum + r_res
            idx = idx +1
        acc_c = corr_sum/idx
        acc_r = r_sum/idx

        total_loss = self.cls_lambda * cls_loss + self.reg_lambda * reg_loss

        return total_loss, cls_loss, reg_loss, acc_c, acc_r