# BoostCamp AI Tech4 Level-2-Data Annotation Project
## Index
* [Project Summary](#project-summary)
* [Dataset Info](#project-summary)
* [Method](#method)

* [Experiments](#experiments)
* [Result](#result)
* [Conclusion](#Conclusion)

## Project Summary

### 학습 데이터 추가 및 수정을 통한 이미지 속 글자 검출 성능 개선 대회
### 기간
- 2022.12.08 ~ 2022.12.15 19:00

### 주제

스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있습니다. 또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있습니다. 이처럼 OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.
![image](https://user-images.githubusercontent.com/83155350/208000638-2462376c-74ff-47ce-9519-735a7df5d49d.png)
OCR task는 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있습니다. 본 대회는 아래와 같은 특징과 제약 사항이 있습니다.

- 본 대회에서는 '글자 검출' task 만을 해결
- Input : 글자가 포함된 전체 이미지
- Output : bbox 좌표가 포함된 UFO Format
## Dataset Info

> [ICDAR 2017](https://rrc.cvc.uab.es/?ch=8)
> [ICDAR 2019](https://rrc.cvc.uab.es/?ch=15)
> - 로그인 후 Multi-script text detection 데이터 다운로드
> - image 파일은 input/data/ICDAR17/images 폴더에 저장
> - ufo 파일은 input/data/ICDAR17/ufo 폴더에 저장

## Method

- [논문](https://arxiv.org/pdf/2108.06949.pdf)을 구현했고 이를 바탕으로 다양한 실험을 진행함
- 논문 내용은 Scene Text Recognition에 특화된 augmentation기법을 제안함
- Scene Text Recognition에 특화된 augmentation기법을 구현하고 1~3개 사이의 augmentation그룹에서 각각 하나씩 임의의 augmentation을 적용함

## Data Augmentation Groups

- [code](https://github.com/boostcampaitech4lv23cv1/level2_dataannotation_cv-level2-cv-06/blob/main/custom_augment.py)

| Warp   | Geometry        | Noise    | Blur     | Weather | Camera           | Process     |
| ------ | --------------- | -------- | -------- | ------- | ---------------- | ----------- |
| Affine | Fliplr          | Gaussian | Gaussian | Rain    | Brightness       | Posterize   |
|        | Flipud          | Shot     | Motion   | Snow    | Contrast         | Solarize    |
|        | Perspective     | Speckle  | Glass    | Fog     | ImageCompression | InvertImg   |
|        | Rotate          | Impulse  | Zoom     | Shadow  |                  | Equalize    |
|        | Fliplr + Flipud |          |          |         |                  | ColorJitter |


## Experiments
### 1) Resize, Random Crop Parameter Search
- 주어진 Baseline에서 input argument를 통해 변경할 수 있는 hyperparameter를 조절하여 성능이 상승할 수 있을지 테스트
- 기존 Baseline에서 조절한 parameter는 2개
  - 입력 이미지의 크기를 조절하는 Resize의 parameter인 image_size
  - Random Crop의 Crop Size를 정하는 input_size

(Public LB Score)
| Image_size                               | Input_size                              | F1                                         | Recall                                     | Precision                                  |
| ---------------------------------------- | --------------------------------------- | ------------------------------------------ | ------------------------------------------ | ------------------------------------------ |
| 1024                                     | 512                                     | 0.4481                                     | 0.3434                                     | 0.6449                                     |
| <span style="color: #ffd33d">1024</span> | <span style="color: #ffd33d">768</span> | <span style="color: #ffd33d">0.4978</span> | <span style="color: #ffd33d">0.3877</span> | <span style="color: #ffd33d">0.6955</span> |
| 1024                                     | 896                                     | 0.4423                                     | 0.3670                                     | 0.5564                                     |
| 1024                                     | 1024                                    | 0.4645                                     | 0.3604                                     | 0.6535                                     |
| 2048                                     | 2048                                    | 0.2915                                     | 0.2011                                     | 0.5301                                     |



### 2) More Datasets
- 학습 Dataset을 추가하면서 모델 성능 비교

(Public LB Score)
| Dataset                                              | F1                                          | Recall                                     | Precision                                  |
| ---------------------------------------------------- | ------------------------------------------- | ------------------------------------------ | ------------------------------------------ |
| Baseline: ICDAR 2017 Korean                          | 0.4481                                      | 0.3434                                     | 0.6449                                     |
| + ICDAR 2017 All                                     | 0.5640                                      | 0.4862                                     | 0.6715                                     |
| <span style="color: #ffd33d">+ ICDAR 2019 All</span> | <span style="color: #ffd33d"> 0.6659</span> | <span style="color: #ffd33d">0.5823</span> | <span style="color: #ffd33d">0.7777</span> |

### 3) More Augmentations
- Baseline code에서 기본적으로 포함된 Augmentation 기법(Resize, RandomCrop) 외에 추가 Augmentation 기법 적용
- 다양한 Augmentaion 기법들을 특성에 따라 group화 ([참고](#data-augmentation-groups))
- 매 입력 이미지 마다 적용할 Augmentation group을 random하게 선택
- 총 group의 수는 1~3개로 지정하여 각각 실험함
- 선택된 각 group에 해당하는 Augmentation 중 하나가 임의로 적용됨
- Crop시 Annotation이 이미지 영역을 벗어나는 문제가 있어 이를 없애고 그 영역을 흰색으로 마스킹하는 방식을 사용

## Results

* [실험 결과](#1-resize-random-crop-parameter-search), Image_size가 1024, Input_size가 768일 때 F1 score가 0.4978로 가장 높았음
* ICDAR 2017, 2019 Dataset으로 학습한 모델의 경우 제출 시 F1 score가 기존 최고 score 대비 0.2 이상 상승함
* 하지만 여기에 다양한 Data Augmentation을 적용한 결과, recall이 0에 근접하여 F1 score도 0.1 이하의 값이 많이 관측됨

## Conclusion
- 논문을 참고하여, 다양한 Augmentation을 통해 Data-Driven한 방법으로 성능을 개선하고자 함
- 하지만 Crop과 Augmentation 사용 시 이미지 영역을 벗어나는 Text를 마스킹 처리한 결과, Text를 포함하지 않는 Negative Sample이 지나치게 많아지게 됨
- 결과적으로, Test Set 이미지에 대한 False Negative가 증가하고 Recall이 매우 낮아진 것으로 보임
## Limitations
- Test set의 특성을 반영한 Validation set을 구성하지 못하여 Public LB 점수로 모델의 성능을 판단함
- code의 module화가 되어있지 않아 debugging 시 원인 분석이 어려웠음
- 논문 내용을 그대로 적용하는 데 집중해서, 본 대회 특성을 고려한 실험을 해보지 못함
  - ex) Crop처럼 특정 Augmentation을 적용하지 않는 실험