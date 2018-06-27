import os

from konlpy.tag import Komoran
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from ctextgen.dataset import Dataset
from ctextgen.model import RNN_VAE

USE_CUDA = torch.cuda.is_available()
GPU_ID = 3

MAX_LENGTH = 30
BATCH_SIZE = 400


def control(input_msg):
    tagger = Komoran()
    
    dataset = Dataset('nsmc/ratings.txt', tagger, max_length=MAX_LENGTH, batch_size=BATCH_SIZE)
    
    Z_DIM = 40
    H_DIM = 300
    C_DIM = 2
    
    model = RNN_VAE(
        dataset.num_words, H_DIM, Z_DIM, C_DIM,
        freeze_embeddings=False,
        gpu=USE_CUDA,
        gpu_id=GPU_ID
    )
    
    test_data = torch.LongTensor(dataset.sentence2idxs(tagger.morphs(input_msg))).unsqueeze(1)
    
    model.load_state_dict(torch.load('models/vae_epoch_300_400.bin'))
    results = model.controlSentence(test_data, t=0.5)
    
    return (dataset.idxs2sentence(results[0], no_pad=True), dataset.idxs2sentence(results[1], no_pad=True))
