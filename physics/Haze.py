import torch
from models import CycleGAN
from deepinv.models.unet import UNet

class Haze():
    def __init__(self, input_shape, blocks, dataset_name, epoch,**kwargs):
        super().__init__(**kwargs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.channel=input_shape[0]
        self.height=input_shape[1]
        self.width=input_shape[2]
        # Initialize generator
        self.G_AB = CycleGAN.GeneratorResNet(input_shape, blocks).to(device)
        self.G_BA = CycleGAN.GeneratorResNet(input_shape, blocks).to(device)
        self.G_AB.eval()
        self.G_BA.eval()
        self.load_ckpt(("saved_models/%s/cholec80_G_AB_%d.pth" % (dataset_name, epoch)), ("saved_models/%s/cholec80_G_BA_%d.pth" % (dataset_name, epoch)))
        for param in self.G_AB.parameters():
            param.requires_grad = False
        for param in self.G_BA.parameters():
            param.requires_grad = False

    def load_ckpt(self, ckpt_path_AB, ckpt_path_BA):
        ckpt_AB = torch.load(ckpt_path_AB, map_location=torch.device('cpu'))
        ckpt_BA = torch.load(ckpt_path_BA, map_location=torch.device('cpu'))
        self.G_AB.load_state_dict(ckpt_AB)
        self.G_BA.load_state_dict(ckpt_BA)

    def A(self, x, **kwargs):
        y=self.G_AB(x)

        return y

    def A_dagger(self, x, **kwargs):

        y=self.G_BA(x)
        return y
