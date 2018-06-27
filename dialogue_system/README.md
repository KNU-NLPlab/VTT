# VTT 1차년도 대화 모델

VTT 과제 1차년도 대화 모델 코드 입니다. 디노이징 메커니즘을 적용한 sequence-to-sequence 모델기반 대화 모델 입니다.
sequence-to-sequence 모델은 opensource framework인 [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)[MIT lisence]를 기반으로 하였습니다.

## Denoising mechanism

디노이징 메커니즘은 노이즈가 가미된 입력 데이터로부터 기존의 정답 데이터를출력하도록 모델을 학습하는 방식이다. 디노이징 메커니즘은 데이터에 노이즈를 가미해 새롭게 생성된 데이터로네트워크를 학습함으로써 데이터 증대(Data augmentation)를 통한 정규화 효과를 얻을 수 있으며 이를 통해 입력을 더 잘 나타내는 강건한 표현을 학습할 수 있다. (김태형, 노윤석, 박성배, 박세영, "한국어 대화 모델 학습을 위한 디노이징 응답 생성", 제29회 한글 및 한국어 정보처리 학술대회)
<p align="center">
    <img src="model.PNG"/>
</p>


## Requirements

```bash
pip install -r requirements.txt
```

## Quickstart

### Step 1: Preprocess the data

```bash
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo
```

data는 입력 발화와 출력 발화가 각각의 source (`src`) and target (`tgt`) 파일에 line단위로 저장하여야하며 각 line은 공백으로 구분되는 token 구성해야 합니다.

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

Validation 파일은 training 과정에서 모델의 convergence를 측정하는데 사용됩니다.


전처리 결과로 아래와 같은 파일들이 생성 됩니다.:

* `demo.train.pt`: serialized PyTorch file containing training data
* `demo.valid.pt`: serialized PyTorch file containing validation data
* `demo.vocab.pt`: serialized PyTorch file containing vocabulary data

모델 내부적으로 각 단어들을 직접적으로 다루지않고 단어들의 index 정보를 이용합니다.

### Step 2: Train the model

```bash
python train.py -data data/demo -save_model demo-model -gpuid 1
```

해당 학습 command는 가장 단순한 예시 입니다. 최소한의 파라미터로 전처리 과정에서 생성한 데이터 파일, 모델을 저장할 path, 학습에 사용할
gpu index를 받습니다(해당 모델 학습을 위해서는 gpu를 반드시 사용하여야 합니다) 이 command를 실행하여 학습하면 500 hidden units의 2-layer LSTM으로 이뤄진 encoder/decoder를 학습합니다.

보다 다양한 파라미터는 `opt.py` 파일을 참조하시기 바랍니다.

### Step 3-1: Evaluation using file

```bash
python translate.py -model demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src data/src-test.txt -output pred.txt -replace_unk -verbose
```

학습한 모델을 평가하는 과정으로 입력 데이터로 txt파일을 사용합니다. output generation에는 beam search가 사용됩니다. 해당 command를 통해 input에
 대한 응답이 `pred.txt`에 저장됩니다.
 
### Step 3-2: Evaluation on shell

```bash
python online_test.py -model demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt 
```

학습한 모델을 파일을 이용해서 평가하지 않고 shell 상에서 console 입력을 통해 평가합니다. 
