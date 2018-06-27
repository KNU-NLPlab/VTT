# coding: utf-8

# In[1]:


import os

from konlpy.tag import Komoran
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from ctextgen.dataset import Dataset
from ctextgen.model import RNN_VAE

torch.__version__

USE_CUDA = torch.cuda.is_available()
GPU_ID = 3

USE_CUDA


# # 1. Get NSMC corpus
# 
# 
# 네이버 영화 관련 코퍼스를 사용해서 학습했습니다. 데이터에 관한 자세한 내용은 밑의 링크를 참조바랍니다.
# 
# 형태소 분석기로는 Komoran을 사용했습니다.
# 
# Naver sentiment movie corpus v1.0 : https://github.com/e9t/nsmc


MAX_LENGTH = 30
BATCH_SIZE = 400

tagger = Komoran()

dataset = Dataset('nsmc/ratings.txt', tagger, max_length=MAX_LENGTH, batch_size=BATCH_SIZE)


# # 3. Train Model

# ## 3.1 Define model 
# 
# 모델은 GRU를 사용한 Variational Recurrent Autoencoder입니다.


Z_DIM = 40
H_DIM = 300
C_DIM = 2


model = RNN_VAE(
    dataset.num_words, H_DIM, Z_DIM, C_DIM,
    freeze_embeddings=False,
    gpu=USE_CUDA,
    gpu_id=GPU_ID
)


def save_base_vae():
    if not os.path.exists('models/'):
        os.makedirs('models/')

    torch.save(model.state_dict(), 'models/vae_epoch_300_complete.bin')


def save_base_vae_iter(iter):
    if not os.path.exists('models/'):
        os.makedirs('models/')

    torch.save(model.state_dict(), 'models/vae_epoch_300_' + str(iter) + '.bin')


# ## 3.2 Train VAE with code
# 
# Positive/Negative 정보와 함께 VAE 학습


NUM_EPOCH = 500
LOG_INTERVAL = 10
SAVE_INTERVAL = 50

LR = 1e-3

kld_start_inc = 350
kld_weight = 0.001
kld_max = 0.1
kld_inc = (kld_max - kld_weight) / (NUM_EPOCH - kld_start_inc)

trainer = optim.Adam(model.vae_params, lr=LR)

for ep in tqdm(range(NUM_EPOCH)):
    # train model
    for inputs, labels in dataset:
        if USE_CUDA:
            inputs = inputs.cuda(GPU_ID)
            labels = labels.cuda(GPU_ID)
            
        recon_loss, kl_loss = model.forward(inputs, labels)
            
        loss = recon_loss + kld_weight * kl_loss
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm(model.vae_params, 5)
        trainer.step()
        trainer.zero_grad()
    
    # anneal kl_weight
    if ep > kld_start_inc and kld_weight < kld_max:
        kld_weight += kld_inc
    
    # print current state
    if ep % LOG_INTERVAL == 0:
        z = model.sample_z_prior(1)
        c = model.sample_c_prior(1)
        _, c_idx = torch.max(c, dim=1)
       
        sample_idxs = model.sample_sentence(z, c)
        sample_sent = dataset.idxs2sentence(sample_idxs)
       
        print('epoch-{}; Loss: {:.4f}; Recon: {:.4f}; KL: {:.4f}; Grad_norm: {:.4f}; Code: {}'
              .format(ep, loss.data[0], recon_loss.data[0], kl_loss.data[0], grad_norm, 'Positive' if c_idx.data[0] == 0 else 'Negative'))
        print('Sample: "{}"'.format(sample_sent))
        print()

    # save current model
    if ep % SAVE_INTERVAL == 0:
        save_base_vae_iter(ep)


save_base_vae()


# ## 3.3 Test model
# 
# 100개의 Test set에 대해서 컨트롤된 문장의 결과를 출력합니다.
# 
# 또한 표준 입력을 받아서 컨트롤된 문장의 결과를 출력합니다.


model = RNN_VAE(
    dataset.num_words, H_DIM, Z_DIM, C_DIM,
    freeze_embeddings=False,
    gpu=USE_CUDA,
    gpu_id=GPU_ID
)


test_set = dataset.getTestData(100)

model.load_state_dict(torch.load('models/vae_epoch_300_400.bin'))
for test in test_set:
    results = model.controlSentence(test[0].unsqueeze(1), t=0.5)
    
    print('Original : ', dataset.idxs2sentence(test[0], no_pad=True))
    print('Positive : ', dataset.idxs2sentence(results[0], no_pad=True))
    print('Negative : ', dataset.idxs2sentence(results[1], no_pad=True))
    print()


tagger = Komoran()

while True:
    sentence = tagger.morphs(input())
    
    if len(sentence) == 0:
        break
    
    sentence = dataset.sentence2idxs(sentence).unsqueeze(dim=1)
    results = model.controlSentence(sentence, t=0.5)
    
    print('Positive : ', dataset.idxs2sentence(results[0], no_pad=True))
    print('Negative : ', dataset.idxs2sentence(results[1], no_pad=True))
    print()
