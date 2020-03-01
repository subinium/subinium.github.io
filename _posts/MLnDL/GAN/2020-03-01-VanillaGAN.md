---
title : "[GAN] Generative Adversarial Networks + Pytorch Code"
category :
  - ML
tag :
  - GAN
  - code
  - review
sidebar_main : true
author_profile : true
use_math : true
header:
  overlay_image : https://images.unsplash.com/photo-1538333702852-c1b7a2a93001?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2252&q=80
  overlay_filter: 0.8
published : true
---

GAN 소스코드로 Pytorch 연습하기

> [Pytorch GAN repo](https://github.com/eriklindernoren/PyTorch-GAN)의 구현체를 사용했습니다.

Ian Goodfellow가 2014년에 발표한 **GAN**은 이미 너무 유명한 논문이고, 이미 다른 글들도 많기에 Pytorch Code를 읽으며 제가 부족한 부분을 정리해봅니다.

> Pytorch 코드를 짜는게 익숙하지 않아 코드를 따라 쓰며, 여러가지 팁들을 익히고 있습니다.

- [Paper](https://arxiv.org/pdf/1406.2661.pdf)
- [Original Implemente](https://github.com/goodfeli/adversarial)

## 기본적인 구조

보통 구현체는 다음과 같은 파일로 이뤄집니다.

- **preprocess** : 파일 호출 및 전처리 
- **utils** : 필요한 함수를 별도로 저장
- **model** : 모델은 class 단위로 설계
- **train** : 모델을 호출하고, 전처리한 데이터셋을 사용하여 

그 외에도 이런 파일들이 있습니다.

- **config.yaml** : 하이퍼파라미터 저장
- **solver** : train을 포함한 여러 과정을 한번에 진행하게 만든 pipeline
- **inference** : 예측 및 생성 모델에서는 별도로 존재하는 경우가 있음

경우에 따라 해당 분류가 여러 개로 나눠질 수도, 합쳐져 있을 수도 있습니다.
위에 소개한 repo는 하나의 파일에 모든 것을 합쳐서 작성되어 있습니다.

## 라이브러리 호출

``` py
import argparse 
import os
import numpy as np
import math

# torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

# torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
```

## Hyperparameter 설정

하이퍼파라미터의 경우에는 다음 방법들이 있습니다.

- `config.yaml` 등의 파일에 저장하여 해당 파일을 불러오는 방법
- cmd에서 argument로 받아오는 방법
    - `argparse` 라이브러리
    - `sys` 라이브러리

취향에 따라 다를 수 있지만, 전체적인 custom이나 실험을 위해서는 `argparse`가 가장 나아보입니다.
해당 argument의 default값과 자료형을 지정할 수 있고, 설명을 추가할 수 있다는 장점이 있습니다

여기 argument를 보면 다음 내용이 필요한 것을 알 수 있습니다.

- `n_epochs` : epoch 수
- `batch_size` : batch 크기
- `lr`, `b1`, `b2` : adam optimzer의 파라미터
- `n_cpu` : cpu threads 수
- `img_size` : 이미지 사이즈
- `channel` : 이미지 채널, Mnist의 경우에는 흑백이미지이므로 1 channel로 사용하기 위함
- `sample_interval` : 제대로 훈련되고 있는지 sample 체크 간격 (출력)

``` py
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False
```

## Generator 

GAN은 Generator와 Discriminator의 구조로 있어 따로 따로 설계하게 됩니다.
각 모델은 `nn.Module`을 상속받습니다.

조금 신기하게 python unpacking을 사용하여 코드를 깔끔하게 만들었습니다. 

사용한 layer는 다음과 같습니다. 여기서 사용하는 layer는 `torch.nn`에 정의되어 있습니다.

- `Linear`
- `BatchNorm1d` : 조금 신기한 것은 BatchNorm1d에 eps 파라미터에 0.8을 넣었다는 점..?
- `LeackyReLU` 
- `Tanh`

구현에 따라 `BatchNorm1d`가 없애거나, `Dropout`층을 추가할 수 있습니다.

그리고 StarGAN 등의 구현체를 보면, init에 layer를 모두 `nn.Sequential`로 쌓고 `forward` 부분을 간소화합니다.

torch에서 `size()`는 numpy의 `shape`과 똑같습니다. 여기서 `size(0)`은 batch_size를 의미합니다.

``` py
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img
```

## Discriminator 

Generator와 유사한 구조(역순)이라는 특징이 있습니다. 

``` py
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
```

## Dataset (DataLoader)

Pytorch는 `DataLoader`를 사용하여 dataset을 만들어 사용합니다.

개인적으로 DataLoader에 대한 이해는 다음 글이 매우 도움되었습니다.

- [How to Build a Streaming DataLoader with PyTorch](https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd)

``` py
# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)
```

## Loss Function & Optimizer
 
손실함수는 Binary Cross Entropy 즉, `BCELoss` 를 사용합니다.

옵티마이저는 adam을 사용합니다.

``` py
# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
```

## Training

훈련은 다음과 같은 과정으로 진행됩니다.

1. epochs * data 만큼 반복문을 돌립니다. data는 batch단위로 iter가 돌아갑니다.
2. valid는 진짜(real)을 의미하는 1, fake는 가짜를 의미하는 0을 의미합니다.
3. ramdom sampling한 tensor인 z를  Generator를 이용하여 이미지를 생성합니다.
4. 생성된 이미지를 Discriminator에 넣어 참/거짓을 분별하고, 이게 얼만나 참인지를 loss로 사용합니다.
5. 실제 이미지와 생성된 이미지를 넣고, 실제 이미지는 1, 생성 이미지는 0으로 구분하도록 계산합니다.
6. 그리고 위의 각각 loss를 합친 것을 discriminator의 loss로 사용합니다.

각 back propagation은 `zero_grad()`, `backward`, `step` 과정으로 이루어지는데 이에 대한 설명은 생략합니다. 

``` py
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
os.makedirs("images", exist_ok=True)

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

```