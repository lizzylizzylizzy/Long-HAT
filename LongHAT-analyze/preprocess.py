
import numpy as np
import pandas as pd

import logging
import wandb
import argparse

from statistics import mean
from tqdm import tqdm

from transformers import RobertaTokenizer

# try to preprocess qlp data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int, default=0, help='gpu id')
    parser.add_argument('--model',type=str,choices={'fast', 'rob', 'r-f', 'f-f'}, required=True, help='model type')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval(default to 500)')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--lr', type=float, default=0.00001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--aml', action="store_true", help='Whether on aml')
    parser.add_argument('--bs', type=int, default=32, help='The batch size (defaults to 32)')
    parser.add_argument('--max_train_length', type=int, default=256, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--segments', type=int, default=4, help='The segments (defaults to 4)')
    parser.add_argument('--epochs', type=int, default=3, help='The number of epochs')
    parser.add_argument('--num_hidden_layers', type=int, default=4, help='The hidden layer (defaults to 4)')
    parser.add_argument('--num_attention_heads', type=int, default=16, help='The attention heads (defaults to 16)')
    parser.add_argument('--hidden_size', type=int, default=256, help='The hidden size (defaults to 256)')
    parser.add_argument('--intermediate_size', type=int, default=1024, help='The intermediate size (defaults to 1024)')
    parser.add_argument('--finetune', action="store_true" , help='Finetune stage or Pretrain')
    parser.add_argument('--ap', action="store_true" , help='attention pooler')
    parser.add_argument('--maxp', action="store_true" , help='maxpooling')
    parser.add_argument('--azure', action="store_true" , help='root_dir')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    # pdb.set_trace()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # logger = logging.getLogger(__name__+args.model)
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    ################################ read config , set log file ################################
 
    max_seq_len = args.max_train_length
    bs = args.bs
    lr = args.lr
    ep = args.epochs
    # model_type =cur_config['model_type']
    # pdb.set_trace()
     
    
    
   
    ################################ text process ################################
    query = []
    text=[]
    label=[]
    
   
    # dataset_path = '/datablob/v-zhuoli2/QL.label.test.tsv'
    train_data_path ='/datablob/v-zhuoli2/QL.label.train.tsv' if args.azure else '../roberta_datasets/ads/QL.label.test.tsv'
    # try_path = "../roberta_datasets/ads/try.tsv"
    train_data = pd.read_csv(train_data_path, delimiter="\t", names=['id', 'label','query', 'doc', 'taskid'], error_bad_lines=False)
    train_data.drop(columns = ['id','taskid'],inplace = True)
    # try_data = pd.read_csv(try_path, sep=',', names=[ 'label','query', 'doc'], error_bad_lines=False)
    # indexs = np.arange(len(try_data))
    len_q = []
    len_d = []
    # for row in tqdm(indexs):
    #     len_q.append(len(try_data.loc[row]['query']))
    #     len_d.append(len(try_data.loc[row]['doc']))
    # print(mean(len_q),max(len_q),min(len_q))
    # print(mean(len_d),max(len_d),min(len_d))
    # dataset_path = '/datablob/v-zhuoli2/QL.log.20m.tsv'
    # pdb.set_trace()
    indexs = np.arange(len(train_data))
    cntt = 0

    news_words = []
    lens_query = []
    lens_doc = []
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base', model_max_len=40000)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', model_max_len=60000)
    tokenizer.model_max_length = 60000
    tokenizer.init_kwargs['model_max_length'] = 60000
    for row in tqdm(indexs):
        # if cntt == 1500:
        #     break
        cntt+=1
        seg_q = tokenizer(train_data['query'][row].split('##!')[0])['input_ids']
        
 
        train_data.at[row,'query'] = seg_q[:int(max_seq_len//4)]

        seg_d = tokenizer((''.join(train_data['doc'][row].split('##!'))))['input_ids']
        train_data.at[row,'doc'] = seg_d[:int(2*max_seq_len)]


        len_q.append(len(train_data.loc[row]['query']))
        len_d.append(len(train_data.loc[row]['doc']))

    train_data.to_csv('/datablob/v-zhuoli2/train20m.tsv',index=False,header=False) if args.azure else train_data.to_csv("../roberta_datasets/ads/try.tsv",index=False,header=False)  
    # train_data.to_csv("../roberta_datasets/ads/try.tsv",index=False,header=False)
    print(mean(len_q),max(len_q),min(len_q))
    print(mean(len_d),max(len_d),min(len_d))
    
    