import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from PIL import Image
from deepinv.models.unet import UNet


import argparse
from skimage.metrics import peak_signal_noise_ratio as skpsnr
from skimage.metrics import structural_similarity as ssim
from Loader import get_dataloader
from torchvision import transforms
import tqdm
import os
import numpy as np
from physics.Haze import Haze
from torchvision.transforms import InterpolationMode
from piq import brisque
import pyiqa

iqa_metric = pyiqa.create_metric('niqe')
physics=Haze(input_shape=[3,256,256], blocks=9, dataset_name="cholec80", epoch=700)
parser = argparse.ArgumentParser(description="Dehaze demo")
parser.add_argument('--model', type=str, default='UNet', help="choose the model")
parser.add_argument('--channel', type=int, default=3, help="choose the input channels")
parser.add_argument('--ckp', default='./ckp/25-02-18-19:51:04_ei_dehaze/ckp_0.pth.tar', type=str, metavar='PATH',
                    help='path to checkpoint of a trained model')
parser.add_argument('--img_height', type=int, default=256, help="choose the input image height")
parser.add_argument('--img_width', type=int, default=256, help="choose the input image height")
parser.add_argument('--output_path', type=str, default='./output_images', help="choose the output path")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
niqe_seq=[]
brisque_seq=[]

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

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    generator = UNet(in_channels=args.channel, out_channels=args.channel, residual=True, scales=5, circular_padding=True, cat=True).to(device)
    checkpoint = torch.load(args.ckp, map_location=device)
    generator.load_state_dict(checkpoint['state_dict'])
    generator.eval()
    transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataloader = get_dataloader(dir_path="cholec80/test", batch_size=1, shuffle=False, num_workers=0,
                                transform=transform)
    for i,x in enumerate(tqdm.tqdm(dataloader)):
        torch.cuda.empty_cache()
        with torch.no_grad():
            generator.eval()
            hazy = x.to(device)
            hazy_dagger = physics.A_dagger(hazy) 
            hazy_dagger = normT(hazy_dagger) * 255
            x1 = generator(normalize(hazy_dagger)) 
            x1 = normT(x1) * 255
            result1 = (x1 / 255 + hazy_dagger / 255) / 2
            combined_image = x1/255
            x1 = x1/255
            x1=torch.clamp(x1, 0, 1)
            combined_image = combined_image.squeeze(0)
            niqe_seq.append(iqa_metric(x1))
            brisque_seq.append(brisque(x1))
            output_path = (f'./output_images/{i}_{iqa_metric(x1):.3f}'
                        f'_{brisque(x1):.3f}.png')
            vutils.save_image(combined_image, output_path)
    print("niqe: ",np.mean([x.cpu().detach().numpy() for x in niqe_seq]))
    print("brisque: ",np.mean([x.cpu().detach().numpy() for x in brisque_seq]))


if __name__ == '__main__':
    main()