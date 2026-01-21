import torch
from sympy.physics.units import degree
from torch.utils.data import DataLoader
import argparse

from ei.ei import EI
from physics.ct import CT
from physics.inpainting import Inpainting
from transforms.rotate import Rotate
from transforms.shift import Shift
from deepinv.transform import shift, rotate, scale
from physics.Haze import Haze
from Loader import get_dataloader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from deepinv.physics.

from deepinv.transform import scale
from deepinv.transform import reflect
from deepinv.transform.projective import Homography
from deepinv.transform.projective import Euclidean
from deepinv.transform.projective import Similarity
from deepinv.transform.projective import Affine
from deepinv.transform.projective import PanTiltRotate

import numpy as np


cuda = torch.cuda.is_available()
parser = argparse.ArgumentParser(description='EI experiment parameters.')

#parser.add_argument('--gpu', default=1, type=int, help='GPU id to use.')
parser.add_argument('--schedule', nargs='+', type=int,default=[3,5,6,7,9,10],
                    help='learning rate schedule (when to drop lr by 10x),'
                         'default [3,5,6,7,9,10] for dehaze')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run ')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate ',
                    dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-8, type=float,
                    metavar='W', help='weight decay (default: 1e-8)',
                    dest='weight_decay')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (1 for dehazing)')
parser.add_argument('--ckp-interval', default=1, type=int)
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# ei specific configs:
parser.add_argument('--ei-trans', default=3, type=int,
                    help='number of transformations for EI (default: 3 for dehaze)')

parser.add_argument('--ei-alpha', default=1, type=float)

parser.add_argument('--contrast-alpha', default=0.1, type=float)

parser.add_argument('--adv_beta', default=1e-8, type=float,
                    help='adversarial strength (default: 1e-8)')

# inverse problem task configs:
parser.add_argument('--task', default='dehaze', type=str,
                    help="inverse problems=['dehaze'] (default: 'dehaze')")


# CycleGAN load parameters
parser.add_argument('--load_epoch', default=700, type=int)
parser.add_argument('--trained_dataset', default='cholec80', type=str)
parser.add_argument('--blocks', default=9, type=int)

#dataset processing
parser.add_argument('--img_height', default=256, type=int)
parser.add_argument('--img_width', default=256, type=int)
parser.add_argument('--channel', default=3, type=int)

def main():
    args = parser.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    #print(device)
    alpha = {'ei': args.ei_alpha, 'contrast': args.contrast_alpha, 'adv': args.adv_beta} # equivariance strength
    lr = {'G': args.lr, 'WD': args.weight_decay}

    assert args.task in ['dehaze']
    if args.task=='dehaze':
        transform = transforms.Compose([
            transforms.Resize((args.img_height, args.img_width),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        input_shape=[args.channel, args.img_height, args.img_width]
        transform_ei = rotate.Rotate(n_trans=1)

        #transform_ei = shift.Shift(n_trans=3)
        #transform_ei = scale.Scale(n_trans=1, factors=[1.12])
        #transform_ei = reflect.Reflect(n_trans=1)
        #transform_ei = Homography(n_trans=1)
        #transform_ei = Similarity(n_trans = 1)
        #transform_ei = Affine(n_trans = 1)
        #transform_ei = PanTiltRotate(n_trans = 1)
        #transform_ei = Euclidean(n_trans = 1)

        dataloader = get_dataloader(dir_path="./cholec80/train",batch_size=1,shuffle=True,num_workers=2,transform=transform)
        physics = Haze(input_shape=input_shape, blocks=args.blocks, dataset_name=args.trained_dataset, epoch=args.load_epoch)
        ei = EI(in_channels=args.channel, out_channels=args.channel,
                img_width=args.img_width, img_height=args.img_height,
                dtype=torch.float, device=device)
    schedule=[5]

    ei.train_ei(dataloader=dataloader,
                physics=physics,
                transform=transform_ei,
                epochs=args.epochs,
                lr=lr,
                alpha=alpha,
                ckp_interval=args.ckp_interval,
                schedule=args.schedule,
                residual=True, pretrained=False, task=args.task, loss_type='l2',
                cat=True, lr_cos=True, report_psnr=True
                )


if __name__ == '__main__':
    main()