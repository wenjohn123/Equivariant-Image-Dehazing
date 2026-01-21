import deepinv as dinv
import torchvision.transforms as transforms
from PIL import Image
import os
import torch
# 读取图片
image_path = "./cholec80/test/hazy/video01_001094.png"  # 请替换为你的图片路径
output_dir = "./transformation_demonstration"
os.makedirs(output_dir, exist_ok=True)

# 读取并转换为tensor
image = Image.open(image_path).convert("RGB")
tensor_image = transforms.ToTensor()(image)
tensor_image=tensor_image.unsqueeze(0)


# 定义变换
transformations = {
    "rotate_180": dinv.transform.rotate.Rotate(n_trans=1, degrees=180),
    "shift": dinv.transform.shift.Shift(n_trans=1),
    "scale_1.12": dinv.transform.scale.Scale(n_trans=1, factors=[1.12]),
    "reflect": dinv.transform.reflect.Reflect(n_trans=1),
    "euclidean": dinv.transform.projective.Euclidean(n_trans=1),
    "similarity": dinv.transform.projective.Similarity(n_trans=1),
    "affine": dinv.transform.projective.Affine(n_trans=1),
    "pan_tilt_rotate": dinv.transform.projective.PanTiltRotate(n_trans=1),
}

# 应用变换并保存
for name, transform in transformations.items():
    transformed_image = transform(tensor_image)
    if name == "scale_1.12":
        transformed_image = torch.clamp(transformed_image, 0, 1)
    print(transformed_image.shape)
    transformed_pil = transforms.ToPILImage()(transformed_image.squeeze(0))
    transformed_pil.save(os.path.join(output_dir, f"{name}.jpg"))

print(f"Transformed images saved in {output_dir}")