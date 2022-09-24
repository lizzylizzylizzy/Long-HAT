import os
import numpy as np
import pickle
import torch
import random
from datetime import datetime
import pdb
import torch.utils.data as torch_data
from tqdm import tqdm

def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def token_id(data, max_len):
    indexs = 0
    token_bs = []
    token2 = np.repeat(1,max_len)
    # 1 -> [sep]
    sep_col2 = np.where(data == 1)[1]
    indexs = np.arange(0,len(sep_col2),2)
    # pdb.set_trace()
    
    for i in list(indexs):

        token1 = np.repeat(0,(sep_col2[i]+1))
        tokens = np.concatenate([token1, token2[len(token1):]],axis=0 )
        token_bs.append(tokens)

    return np.array(token_bs,dtype='int32') 

def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
    
    if not torch.cuda.is_available():
        device = torch.device('cpu')
        return device
    
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]



class SuperviseSampleDataset(torch_data.Dataset):
    def __init__(self,df, tokenizer,type,max_seq_len,segments = None) -> None:
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.type = type
        self.msl = max_seq_len
        self.segs = segments

    def __getitem__(self, index):
        cur_row = self.df.loc[index]
        label = cur_row['label']
        q = self.tokenizer(cur_row['query'].split('##!')[0])['input_ids']
        # d = self.tokenizer((''.join(cur_row['doc'].split('##!'))))['input_ids']
        d = self.tokenizer((''.join(cur_row['doc'])))['input_ids']
        pad_id = self.tokenizer('<pad>')['input_ids'][1]
        sep_id = self.tokenizer("</s>")['input_ids']
        if self.type == 'rob' or self.type == 'fast':
            q=q[:int(self.msl//2)] + sep_id
            content = (q+d)[:self.msl] 
            content = content + [pad_id]*(self.msl-len(content))

            return content, label
        else:
            seg_len = self.msl // self.segs
            content =[]
            q = q[:int(seg_len//2)]  + sep_id
            
            seg_text=d[:(self.msl-self.segs*len(q))]
            seg_start = 0
            seg_end = seg_len-len(q)
            for cnt in range(self.segs):
                if seg_end <= len(seg_text) :
                    content += (q+seg_text[seg_start:seg_end])
                elif seg_start <= len(seg_text):
                    content += (q+seg_text[seg_start:]+[pad_id]*(seg_end - len(seg_text)))
                else:
                    content += ([pad_id]*(seg_len))
                seg_start = seg_end
                seg_end += seg_len-len(q)
            if(len(content) > self.msl):
                pdb.set_trace()
            return content, label

    
    def __len__(self):
        return len(self.df)
    
def supervise_sample_collate_fn(batch):
    

    label = [tple[1] if tple[1] is not None else None for tple in batch] 
    label = torch.LongTensor(label)

  
    data = [tple[0] for tple in batch]
    data = torch.LongTensor(data)


    return data, label

