---
title : "[Pytorch] DataLoader parameter별 용도"
category :
  - ML
tag :
  - pytorch
  - dataloader
  - parameter
  - sampler
  - num_workers
  - pin_memory
  - collate_fn
sidebar_main : true
author_profile : true
use_math : true
header:
  overlay_image : https://pytorch.org/tutorials/_static/img/thumbnails/cropped/Introduction-to-TorchScript.png
  overlay_filter: 0.5
published : true
---
pytorch reference 문서를 다 외우면 얼마나 편할까!!

PyTorch는 `torch.utils.data.Dataset`으로 Custom Dataset을 만들고, `torch.utils.data.DataLoader`로 데이터를 불러옵니다. 

하지만 하다보면 데이터셋에 어떤 설정을 주고 싶고, 이를 조정하는 파라미터가 꽤 있다는 걸 알 수 있습니다.
그래서 이번에는 torch의 `DataLoader`의 몇 가지 기능을 살펴보겠습니다.

## DataLoader Parameters

### dataset

- *`Dataset`*

`torch.utils.data.Dataset`의 객체를 사용해야 합니다. 

참고로 torch의 `dataset`은 2가지 스타일이 있습니다.

- **Map-style dataset** 
  - index가 존재하여 data[index]로 데이터를 참조할 수 있음
  - `__getitem__`과 `__len__` 선언 필요
- **Iterable-style dataset** 
  - random으로 읽기에 어렵거나, data에 따라 batch size가 달라지는 데이터(dynamic batch size)에 적합
  - 비교하자면 stream data, real-time log 등에 적합
  - `__iter__` 선언 필요

이 점을 유의하며 아래의 파라미터 설명을 읽으면 더 이해가 쉽습니다.

### batch_size

- *`int`, optional, default=`1`*

**배치(batch)**의 크기입니다. 데이터셋에 50개의 데이터가 있고, batch_size가 10라면 총 50/10=5, 즉 5번의 iteration만 지나면 모든 데이터를 볼 수 있습니다.

이 경우 반복문을 돌리면 `(batch_size, *(data.shape))`의 형태의 `Tensor`로 데이터가 반환됩니다. dataset에서 return하는 모든 데이터는 Tensor로 변환되어 오니 Tensor로 변환이 안되는 데이터는 에러가 납니다.

### shuffle

- *`bool`, optional, default=`False`*

데이터를 DataLoader에서 섞어서 사용하겠는지를 설정할 수 있습니다.
실험 재현을 위해 `torch.manual_seed`를 고정하는 것도 포인트입니다.

> 그냥 Dataset에서 initialize할 때, random.shuffle로 섞을 수도 있습니다.

### sampler

- *`Sampler`, optional*

`torch.utils.data.Sampler` 객체를 사용합니다.

sampler는 index를 컨트롤하는 방법입니다. 데이터의 index를 원하는 방식대로 조정합니다.
즉 index를 컨트롤하기 때문에 설정하고 싶다면 `shuffle` 파라미터는 `False`(기본값)여야 합니다.

map-style에서 컨트롤하기 위해 사용하며 `__len__`과 `__iter__`를 구현하면 됩니다.
그 외의 미리 선언된 Sampler는 다음과 같습니다.

- `SequentialSampler` : 항상 같은 순서
- `RandomSampler` : 랜덤, replacemetn 여부 선택 가능, 개수 선택 가능
- `SubsetRandomSampler` : 랜덤 리스트, 위와 두 조건 불가능
- `WeigthRandomSampler` : 가중치에 따른 확률
- `BatchSampler` : batch단위로 sampling 가능
- `DistributedSampler` : 분산처리 (`torch.nn.parallel.DistributedDataParallel`과 함께 사용)

### batch_sampler

- *`Sampler`, optional*

위와 거의 동일하므로 생략합니다.

### num_workers

- *`int`, optional, default=`0`*

데이터 로딩에 사용하는 subprocess개수입니다. (멀티프로세싱)

기본값이 0인데 이는 data가 main process로 불러오는 것을 의미합니다.
그럼 많이 사용하면 좋지 않은가? 라고 질문하실 수도 있습니다.

하지만 데이터를 불러 CPU와 GPU 사이에서 많은 교류가 일어나면 오히려 병목이 생길 수 있습니다.
이것도 trade-off관계인데, 이와 관련하여는 다음 글을 추천합니다.

- [DataLoader num_workers에 대한 고찰](https://jybaek.tistory.com/799)

### collate_fn

- *callable, optional*

map-style 데이터셋에서 sample list를 batch 단위로 바꾸기 위해 필요한 기능입니다.
zero-padding이나 Variable Size 데이터 등 데이터 사이즈를 맞추기 위해 많이 사용합니다.

### pin_memory

- *`bool`, optional*

`True`러 선언하면, 데이터로더는 Tensor를 CUDA 고정 메모리에 올립니다.

어떤 상황에서 더 빨라질지는 다음 글을 참고합시다.

- discuss.Pytorch : [When to set pin_memory to true?](https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723)

### drop_last

- *`bool`, optional*

`batch` 단위로 데이터를 불러온다면, batch_size에 따라 마지막 batch의 길이가 달라질 수 있습니다.
예를 들어 data의 개수는 27개인데, batch_size가 5라면 마지막 batch의 크기는 2가 되겠죠.

batch의 길이가 다른 경우에 따라 loss를 구하기 귀찮은 경우가 생기고, batch의 크기에 따른 의존도 높은 함수를 사용할 때 걱정이 되는 경우 마지막 batch를 사용하지 않을 수 있습니다.

### time_out

- *numeric, optional, default=`0`*

양수로 주어지는 경우, DataLoader가 data를 불러오는데 제한시간입니다.

### worker_init_fn

- *callable, optional, default='None'*

num_worker가 개수라면, 이 파라미터는 어떤 worker를 불러올 것인가를 리스트로 전달합니다.

> 아래 2개는 언제 사용하는걸까요?

## Reference

- official : [torch.utils.data](https://pytorch.org/docs/stable/data.html)

- Hulk의 개인 공부용 블로그 : [pytorch dataset 정리](https://hulk89.github.io/pytorch/2019/09/30/pytorch_dataset/) : 핵심적인 함수의 사용법들과 커스텀 클래스 선언이 궁금하신 분들에게 추천합니다.