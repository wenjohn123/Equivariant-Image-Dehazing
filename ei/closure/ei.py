import torch
import numpy as np
from deepinv.utils.metric import cal_psnr, cal_mse
from fontTools.ttLib.tables.G_D_E_F_ import table_G_D_E_F_
from numpy.ma.core import shape, negative
from scipy.special.cython_special import sph_harm
from torch.distributions.constraints import positive

from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
from models import CycleGAN
from models.discriminator import Discriminator
from piq import brisque
import pyiqa
import os

iqa_metric = pyiqa.create_metric('niqe')



def normT(tensor):
    B, C, H, W = tensor.shape
    normalized_tensor = []
    for index in range(B):
        img = tensor[index]
        min_val = img.min()
        max_val = img.max()
        normalized_tensor.append((img - min_val) / (max_val - min_val))

    return torch.stack(normalized_tensor, dim=0)


def closure_ei(net, dataloader, physics, transform,
               optimizer, criterion_mc, criterion_ei,
               alpha, dtype, device, reportpsnr=False,
               ):
    loss_mc_seq, loss_ei_seq, loss_seq, niqe_seq, mse_seq, psnr_physic = [], [], [], [], [], []
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    for i, x in enumerate(tqdm(dataloader)):
        
        hazy = x.to(device)
        hazy_dagger = physics.A_dagger(hazy)
        hazy_dagger = normT(hazy_dagger) * 255
        x1 = net(hazy_dagger)
        x1 = normT(x1) * 255
        result1 = (x1 / 255 + hazy_dagger / 255) / 2
        hazy_dagger1 = normalize(result1)
        y1 = physics.A(hazy_dagger1)
        x2_clear = transform(result1)
        x2 = normalize(x2_clear)
        x2 = physics.A(x2)
        x2 = normalize(x2)
        x2_dagger = physics.A_dagger(x2)
        x2_dagger = normT(x2_dagger) * 255
        x3 = net(x2_dagger)
        x3 = normT(x3) * 255
        result2 = (x3 / 255 + x2_dagger / 255) / 2

        loss_mc = criterion_mc(hazy, y1)
        loss_ei = criterion_ei(result2, x2_clear)

        loss = loss_mc + alpha['ei'] * loss_ei

        loss_mc_seq.append(loss_mc.item())
        loss_ei_seq.append(loss_ei.item())
        loss_seq.append(loss.item())

        if reportpsnr:
            x1 = x1 / 255
            niqe_seq.append(iqa_metric(x1))
            mse_seq.append(brisque(x1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_closure = [np.mean(loss_mc_seq), np.mean(loss_ei_seq), np.mean(loss_seq)]

    if reportpsnr:
        mse_seq_np = [0]

        loss_closure.append(np.mean([x.detach().cpu().numpy() for x in niqe_seq]))
        loss_closure.append(np.mean([x.detach().cpu().numpy() for x in mse_seq]))

    return loss_closure





