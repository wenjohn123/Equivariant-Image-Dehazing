# Equivariant Learning for Unsupervised Image Dehazing

📄 **Paper**:  
**Equivariant Learning for Unsupervised Image Dehazing**  
Zhang Wen\*, Jiangwei Xie\*, Dongdong Chen  
\* Equal contribution  

📬 **Corresponding Author**:  
Dongdong Chen (d.chen@hw.ac.uk)

📎 **arXiv**:
https://arxiv.org/abs/2601.13986 

![pipeline](./images/Figure_3.pdf)
## Abstract

Image Dehazing (ID) aims to produce a clear image from an observation contaminated by haze. Current ID methods typically rely on carefully crafted priors or extensive haze-free ground truth, both of which are expensive or impractical to acquire, particularly in the context of scientific imaging. We propose a new unsupervised learning framework called Equivariant Image Dehazing (EID) that exploits the symmetry of image signals to restore clarity to hazy observations. By enforcing haze consistency and systematic equivariance, EID can recover clear patterns directly from raw, hazy images. Additionally, we propose an adversarial learning strategy to model unknown haze physics and facilitate EID learning. Experiments on two scientific image dehazing benchmarks (including cell microscopy and medical endoscopy) and on natural image dehazing have demonstrated that EID significantly outperforms state-of-the-art approaches. By unifying equivariant learning with modelling haze physics, we hope that EID will enable more versatile and effective haze removal in scientific imaging.

---

## Environment Setup

Recommended **Python = 3.10** and **Pytorch=2.2.0** 

### Required Dependencies

```text
pytorch==2.2.0
deepinv==0.3.0
pyiqa==0.1.14.1
opencv-python==4.12.0
numpy==1.26.0
```
### Installation Example
```
conda create -n share python=3.10 -y
conda activate share
```
Install pytorch=2.2.0, cuda 12.1 version from github and install other essential resources
```
pip install deepinv==0.3.0
pip install pyiqa==0.1.14.1
pip install opencv-python==4.12.0
pip install numpy==1.26.0
```
## Dataset Preparation
We provide the original datasets sources and our cleaned datasets will be released soon.
You can download the Cholec80 dataset from [Cholec80 Dataset](https://camma.unistra.fr/datasets/), the Cell97 dataset from [Hand-segmented 2D Nuclear Images](https://murphylab.web.cmu.edu/data/2009_ISBI_Nuclei.html), and the RESIDE series datasets from [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-β).

After downloading, place the data under ```./data```

## Training 

### Train Dehazing task
```
python demo_train.py
```

## Testing
```
python demo_dehaze.py
```
## Results

Some of the comparing results are as follows.
![results1](./images/Figure_7.pdf)


## Citation
If you find this work useful, please cite:

```
@misc{wen2026equivariantlearningunsupervisedimage,
      title={Equivariant Learning for Unsupervised Image Dehazing}, 
      author={Zhang Wen and Jiangwei Xie and Dongdong Chen},
      year={2026},
      eprint={2601.13986},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.13986}, 
}
```

## Acknowledgements
Most of our code is build upon [deepinverse](https://github.com/deepinv/deepinv), thanks for their framework
