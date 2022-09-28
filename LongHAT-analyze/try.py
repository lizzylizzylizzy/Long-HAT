# from transformers import RobertaTokenizer, RobertaModel
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# # model = RobertaModel.from_pretrained('roberta-base')
# # text = "Replace <pad> <mask> me by any text you'd like. <s>"
# text = "Replace me by any text you'd like. <pad> <mask>"
# encoded_input = tokenizer(text, return_tensors='pt')
# import pdb; pdb.set_trace()
# output = model(**encoded_input)

# s = [1, 2, 3, 4, 5]
# e = enumerate(s)
# for index, value in e:
#     print('%s, %s' % (index, value))
# print(index)

import torch
import pandas as pd
import torch.nn.functional as F
from datasets import load_dataset
from accelerate import Accelerator
import numpy as np
import pandas as pd
import os
import logging
import wandb
import argparse
import datetime
import pdb
import json
import torch
import deepspeed
import torch.nn as nn
import torch.optim as optim
import time
from statistics import mean
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import *
from pytorch_transformers import RobertaConfig
from transformers import RobertaTokenizer
from accelerate import Accelerator

from datasets import load_dataset
from nltk.tokenize import wordpunct_tokenize
from models.combined import Roberta, CombinedRoberta
from utils import init_dl_program, supervise_sample_collate_fn, SuperviseSampleDataset
from accelerate import DistributedDataParallelKwargs
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
# accelerator = Accelerator()
# - device = 'cpu'
device = accelerator.device
 
#script to analyze data distribution

# dataset = load_dataset("imdb")
# data = torch.utils.data.DataLoader(dataset, shuffle=True)
query = []
text=[]
label=[]
dataset_path = '../roberta_datasets/ads/QL.label.test.tsv'

data = pd.read_csv(dataset_path, delimiter="\t", names=['id', 'label','query', 'doc', 'taskid'], error_bad_lines=False)

indexs = np.arange(len(data))
cntt = 0
offset = 0
max_seq_len  = 512
news_words = []
lens_query = []
lens_doc = []
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', model_max_len=40000)
for row in tqdm(indexs):
    if cntt == 1500:
        break
    cntt+=1
    seg_q = tokenizer(data['query'][row+offset].split('##!')[0])['input_ids']
    # pdb.set_trace()
    seg_d = tokenizer((''.join(data['doc'][row+offset].split('##!'))))['input_ids']
    # pdb.set_trace()
    if len(seg_d) <=3 :
        continue
    query.append(seg_q)
    text.append(seg_d)
    label.append(data['label'][row+offset])
    lens_query.append(len(seg_q))
    lens_doc.append(len(seg_d))
    
   
    sample= (seg_q + seg_d)[:(max_seq_len)]
    news_words.append(sample+[tokenizer('<pad>')['input_ids'][1]]*(max_seq_len-len(sample)))
        

        
print('query avg: {}, max: {}, min: {}'.format(mean(lens_query), max(lens_query), min(lens_query)))

print('text avg: {}, max: {}, min: {}'.format(mean(lens_doc), max(lens_doc), min(lens_doc)))

    
news_words=np.array(news_words,dtype='int32') 

label=np.array(label,dtype='int32') 
label[label>=1]=1


index=np.arange(len(label))


train_index=index[:int(news_words.shape[0]*0.8)]
valid_index=index[int(news_words.shape[0]*0.8):]
print("All train data amount: ", news_words[train_index].shape)

print("All valid data amount: ", news_words[valid_index].shape)
train_dataset = SuperviseSampleDataset(news_words[train_index], label[train_index])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=supervise_sample_collate_fn, drop_last=True)

valid_dataset = SuperviseSampleDataset(news_words[valid_index], label[valid_index])
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=supervise_sample_collate_fn, drop_last=False)
 
config=RobertaConfig.from_json_file('roberta.json')
# model = torch.nn.Transformer().to(device)
# optimizer = torch.optim.Adam(model.parameters())
model = Roberta(config, finetune=False, use_ap = False, mask_id = tokenizer('<mask>')['input_ids'][1]).to(device)
optimizer = torch.optim.Adam(model.parameters())
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
 
 
model.train()
for epoch in range(5):
    loss = 0.0
    for source, targets in train_loader:
        source = source.to(device)
        targets = targets.to(device)
        # import pdb;pdb.set_trace()


        optimizer.zero_grad()


        bz_loss, _ = model(source,targets)
        loss += bz_loss.data.float()
        unified_loss=bz_loss
        # loss = F.cross_entropy(output, targets)
        print(loss)

        accelerator.backward(unified_loss)
# -         loss.backward()


        optimizer.step()