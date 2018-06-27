import os
import re
import random
import pickle

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

from operator import itemgetter
from tqdm import tqdm


class Dataset(Dataset):
    
    def __init__(self, file_path, tagger, num_words=30000, max_length=15, batch_size=512):
        self.FILENAME = file_path
        self.DATA_PICKLE = 'pickle/DATA.pkl'
        self.VOCAB_PICKLE = 'pickle/VOCAB.pkl'
        
        self.max_length = max_length
        self.batch_size = batch_size
        
        self.idx2word = ['<unk>', '<pad>', '<s>', '</s>']
        self.word2idx = {word:idx for idx, word in enumerate(self.idx2word)}
        
        self.num_words = num_words
        self.batch_data = self.getBatchData(tagger)
        
    
    def __len__(self):
        return len(self.batch_data)

    def __getitem__(self, idx):
        return self.batch_data[idx]
    
    def idxs2sentence(self, idxs, no_pad=False):
        return ' '.join([self.idx2word[i] for i in idxs if no_pad is False or (no_pad is True and self.idx2word[i] != '<pad>')])
    
    def sentence2idxs(self, seq):
        idxs = list(map(lambda w: self.word2idx[w] if w in self.word2idx.keys() else self.word2idx["<unk>"], seq))
        tensor = torch.LongTensor(idxs)
        return tensor
    
    def getTestData(self, num):
        return random.sample(self.test_data, k=num)
    
    def getBatchData(self, tagger):
        if not os.path.exists('pickle/'):
            os.makedirs('pickle/')
            
        if os.path.isfile(self.DATA_PICKLE) and os.path.isfile(self.VOCAB_PICKLE): 
            print('Loading stored data pickle...')
            
            with open(self.DATA_PICKLE, 'rb') as fp:
                train = pickle.load(fp)
            with open(self.VOCAB_PICKLE, 'rb') as fp:
                self.idx2word = pickle.load(fp)
            print('{} sentences can be used'.format(len(train)))
                
            self.word2idx = {word:idx for idx, word in enumerate(self.idx2word)}
                       
        else:
            print('Preprocessing data...')            
            
            # read raw datas
            raw_data = open(self.FILENAME,'r',encoding='utf-8').readlines()[1:]
            raw_data = [[d.split('\t')[1],d.split('\t')[2][:-1]] for d in raw_data]
            print('{} sentences are loaded'.format(len(raw_data)))
        
            # filter something like special symbols
            filters = re.compile('[^ ㄱ-ㅣ가-힣0-9]+')
            
            # tokenize sentences
            train = []
            word_with_cnt = []
            for d in tqdm(raw_data):
                d[0] = filters.sub('', d[0])
                if len(d[0]) <= 1:
                    continue
                    
                token = tagger.morphs(d[0])
                
                pad_num = self.max_length - len(token)
                if pad_num < 0:
                    continue
                    
                # append token to vocab_dictionary
                for t in token:
                    # if the length of token is bigger than 5, consider it UNK_WORD
                    if len(t) > 5:
                        continue
                    idx = next((i for i in range(len(word_with_cnt)) if word_with_cnt[i][0] == t), None)
                    if idx == None:
                        word_with_cnt.append([t, 1])
                    else:
                        word_with_cnt[idx][1] += 1
                
                token.insert(0, "<s>")
                token.append("</s>")
                token += ["<pad>"]*pad_num
                    
                train.append([token, d[1]])
            
            random.shuffle(train)
                       
            print('{} sentences can be used'.format(len(train)))
            
            self.idx2word += [e[0] for e in sorted(word_with_cnt, key=itemgetter(1), reverse=True)]
            print('The number of original vocab is ', len(self.idx2word))
            
            self.idx2word = self.idx2word[:self.num_words]
            self.word2idx = {word:idx for idx, word in enumerate(self.idx2word)}
            
            print('Saving data and vocab pickle...')
            
            with open(self.DATA_PICKLE, 'wb') as fp:
                pickle.dump(train, fp)
            with open(self.VOCAB_PICKLE, 'wb') as fp:
                pickle.dump(self.idx2word, fp)
                
        print('Success to load data')    
        
        train_len = int(len(train)*0.8)
        train_data = [[self.sentence2idxs(d[0]), d[1]] for d in train[:train_len]]
        self.test_data = [[self.sentence2idxs(d[0]), d[1]] for d in train[train_len:]]
            
        num = len(train_data)
        batch_datas = []
        for idx in range(0, num, self.batch_size):
            batch_data = train_data[idx:min(idx + self.batch_size, num)]
            
            """
            batch_x : Original word sequence
            batch_y : Code model wants to control
            """
            batch_x = Variable(torch.stack([x[0] for x in batch_data], dim=1))
            batch_y = Variable(torch.FloatTensor([[1, 0] if int(x[1]) == 1 else [0, 1] for x in batch_data]))
            batch_datas.append([batch_x, batch_y])
        
        return batch_datas
