# VTT 1차년도 문장 속성 컨트롤 모델

VTT 과제 1차년도 문장 속성 컨트롤 모델 코드입니다.

Variational Autoencoder를 통한 문장 속성 학습 모델입니다.

일반적인 VAE가 생성하는 Latent representation z와 컨트롤 하고자 하는 속성 c를 분리시켜,

하나의 문장에 대해서 z와 c로 표현하도록 하여 Decoding해서 Controlled Sentence를 얻게 됩니다.

- z : Latent representation
- c : Controllable attribute 


## Model Architecture
![1](https://user-images.githubusercontent.com/21354982/41916490-1e1ffeea-7993-11e8-89f6-2b048ab07614.png)

Input 𝑥 :
- 많은 양의 unlabeled sentences
- 소수의 labeled sentences
     
Output 𝑥' :
- 컨트롤 된 속성에 따르는 Sentences



## Requirements
```
pip install -r requirements.txt
```

## Quickstart
```
python train_model.py
```


## Citation
```
@inproceedings{hu2017toward,
  title={Toward controlled generation of text},
  author={Hu, Zhiting and Yang, Zichao and Liang, Xiaodan and Salakhutdinov, Ruslan and Xing, Eric P},
  booktitle={International Conference on Machine Learning},
  pages={1587--1596},
  year={2017}
}
```
