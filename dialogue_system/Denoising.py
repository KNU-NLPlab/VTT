# -*- coding : utf-8 -*-
# Denoising.py
# by thkim

# Code for implementing denosing mechanism related things

# Refer to readme.md about denoising mechanism
import torch
import torch.nn
import random

class Denoising():
  """
  Implements noise function used in denoisingmechanism for pytorch

  Args:
    noising_prob (:float): probability of noise

  """
  def __init__(self, noising_prob=0.1):
    self.noising_prob = noising_prob


  def noise_src(self, src, src_lengths):
    """
    Inject noise into src data and return noised data
    noise scheme is "drop+swap"

    Args:
      src (:obj Tensor (src_length * batchsize * 1)):
        original source sequence 
      src_lengths (:obj LongTensor) (batchsize)):
        sequence length of each elements in batch

    Returns:
      (:obj 'Float Tensor'(noised_src_length * batchsize * 1), :obj 'Long Tensor(batchsize)'
        * noised src sequence
        * nosied src lengths    
    
    """
    noise = [] # index of injecting noise
    p = self.noising_prob

    # drop noise
    for i in range(min(src_lengths)):
      if min(src_lengths) - len(noise) < 2:
        break
      if random.random() <= 1-p:
        noise.append(i)
                        
    # swap noise
    if len(noise) > 1:
      for i in range(len(noise)-1):
        if random.random() <= p:
          tmp = noise[i]
          noise[i] = noise[i+1]
          noise[i+1] = tmp
          i = i + 1
                        
    if len(noise) > 0:
      # assume training is processed on the gpu
       noised_src = src.index_select(0, torch.autograd.Variable(torch.LongTensor(noise)).cuda() ) # make new tensor
       num_noised = src.size()[0]-len(noise)
                    
       noised_src_lengths = src_lengths - (src.size()[0]-len(noise))
       noised_src_lengths[noised_src_lengths < num_noised] = num_noised
    else:
      noised_src = src
      noised_src_lengths = src_lengths

    return noised_src, noised_src_lengths

  
  
    
