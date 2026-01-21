import torch

from torchvision.utils import save_image

from utils.metric import cal_psnr, cal_ssim

import argparse
from skimage.metrics import peak_signal_noise_ratio as skpsnr
from Loader import get_dataloader
from torchvision import transforms
import tqdm
import os
import numpy as np
from physics.Haze import Haze
import torch.nn.functional as F

physics=Haze(input_shape=[3,256,256], blocks=9, dataset_name="cholec80", epoch=480)

parser = argparse.ArgumentParser(description="Test CycleGAN Inverse Ability")
parser.add_argument('--model', type=str, default='UNet', help="choose the model")

parser.add_argument('--channel', type=int, default=3, help="choose the input channels")

parser.add_argument('--ckp', default='./ckp/net/ckp_2.pth.tar', type=str, metavar='PATH',
                    help='path to checkpoint of a trained model')

parser.add_argument('--img_height', type=int, default=256, help="choose the input image height")
parser.add_argument('--img_width', type=int, default=256, help="choose the input image height")

parser.add_argument('--output_path', type=str, default='./output_images', help="choose the output path")
device = 'cuda' if torch.cuda.is_available() else 'cpu'


mse=[]


def normT(tensor):
    B, C, H, W = tensor.shape
    normalized_tensor = []
    for index in range(B):
        img = tensor[index]
        min_val = img.min()
        max_val = img.max()
        normalized_tensor.append((img - min_val) / (max_val - min_val))

    return torch.stack(normalized_tensor, dim=0)


def main():
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataloader = get_dataloader(dir_path="./gradient/", batch_size=1, shuffle=False, num_workers=0,
                                transform=transform)

    for i, x in enumerate(tqdm.tqdm(dataloader)):
       # clear=x[0].to(device)
        hazy=x.to(device)

        #hazy_dagger=physics.A_dagger(hazy)
        hazy_2=physics.A(hazy)
        hazy_2=normT(hazy_2)
        filename = f"gradient/add_hazy_result/{i}.png"
        save_image(hazy_2, filename)


        #mse.append(F.mse_loss(hazy, hazy_2))

    #print("MSE mean: ", np.mean(mse))

if __name__ == '__main__':
    main()





# Test MSE mean:  0.019145634
# Train MSE mean:  0.018066684