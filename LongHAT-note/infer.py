import numpy as np
import pandas as pd
import os
import logging
# import wandb
import argparse
import datetime
import pdb
import json
import torch
# import deepspeed
import torch.nn as nn
import torch.optim as optim
# from torchstat import stat
# from ptflops import get_model_complexity_info
import time
from statistics import mean
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import *
from pytorch_transformers import RobertaConfig
from transformers import RobertaTokenizer, RobertaModel
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
# from datasets import load_dataset
# from nltk.tokenize import wordpunct_tokenize
from models.combined import Roberta, CombinedRoberta, Evaluate
from utils import init_dl_program, supervise_sample_collate_fn, SuperviseSampleDataset
# import global_time 
# import copy


 
def statistic_arr(myarr, bins):
    """
     print the data distribution in myarr based on the segment in bins
     Args:
        myarr (np.array):  the data for statistic
        bins (np.array):  the segment of the data, such as[0.5, 1, 5, 10]
    """
    myarr = np.array(myarr)
    statis= np.arange(bins.size)
    result = []
    for i in range(0, bins.size):
        statis[i]= myarr[myarr < bins[i]].size
        str_item = ("data<" + str(bins[i]) , str(round(statis[i]/myarr.size,5) ))
        result.append(str_item)
    
    print(result)

def tester(model, loader, accelerator, logger, epoch, wandb=None, mode=None):
    allpred_test=[]
    alltrue_test=[]
    tester_loss = 0.0
    t_inf_start = time.time()
    for iter_index , (log_ids, targets) in enumerate(loader, 1):   

        log_ids= log_ids.to(accelerator.device)
        targets= targets.to(accelerator.device)
        with torch.no_grad():
            bz_loss3, y_hat3 = model(log_ids,targets)
        
        loss_gatherd2, pred_gathered2, labels_gathered2 = accelerator.gather((bz_loss3,y_hat3, targets)) 

        allpred_test+=pred_gathered2.to('cpu').detach().numpy().tolist()
        alltrue_test+=labels_gathered2.to('cpu').detach().numpy().tolist()

        tester_loss += loss_gatherd2.sum().data.float()

    
    m = nn.Softmax(dim=1)
    allpred_test_np = m(torch.tensor(allpred_test)).numpy()

    allpred_test_well = 1-allpred_test_np[:,0]
    y_true = np.array(alltrue_test)
    y_true[y_true>1] = 1
    test_auc_score = roc_auc_score(y_true, allpred_test_well)

    tester_loss = tester_loss.data / iter_index
    cur_inf_time = time.time() - t_inf_start

    if torch.distributed.get_rank() == 0:
        print(' Epoch: {},  {} loss: {:.5f}'.format(epoch, mode, tester_loss.item()))
        logger.info( ' Epoch: {},  {} loss: {:.5f}'.format(epoch, mode, tester_loss.item()))
        print(' Epoch: {},  {} auc: {:.5f}'.format(epoch, mode, test_auc_score))
        logger.info( ' Epoch: {},  {} auc: {:.5f}'.format(epoch, mode, test_auc_score))

        logger.info("Epoch {} inference time, {} seconds".format(epoch, datetime.timedelta(seconds=cur_inf_time)))
        print("Epoch {} inference time, {} seconds".format(epoch, datetime.timedelta(seconds=cur_inf_time)))
    return cur_inf_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int, default=0, help='gpu id')
    parser.add_argument('--model',type=str,choices={'fast', 'rob', 'r-f', 'f-f'}, required=True, help='model type')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval(default to 100)')
    parser.add_argument('--save_interval', type=int, default=100, help='Save checkpoint interval(default to 100)')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--lr', type=float, default=0.00001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--drop', type=float, default=None, help='The learning rate (defaults to 0.1)')
    parser.add_argument('--aml', action="store_true", help='Whether on aml')
    parser.add_argument('--bs', type=int, default=32, help='The batch size (defaults to 32)')
    parser.add_argument('--max_train_length', type=int, default=256, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--segments', type=int, default=4, help='The segments (defaults to 4)')
    parser.add_argument('--epochs', type=int, default=3, help='The number of epochs')
    parser.add_argument('--num_hidden_layers', type=int, default=4, help='The hidden layer (defaults to 4)')
    parser.add_argument('--layers', type=str, default=None, help='Layers for Long-HAT')
    parser.add_argument('--num_attention_heads', type=int, default=16, help='The attention heads (defaults to 16)')
    parser.add_argument('--hidden_size', type=int, default=256, help='The hidden size (defaults to 256)')
    parser.add_argument('--intermediate_size', type=int, default=1024, help='The intermediate size (defaults to 1024)')
    parser.add_argument('--finetune', action="store_true" , help='Finetune stage or Pretrain')
    parser.add_argument('--ap', action="store_true" , help='attention pooler')
    parser.add_argument('--maxp', action="store_true" , help='maxpooling')
    parser.add_argument('--azure', action="store_true" , help='root_dir')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    

    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    logger.setLevel(level = logging.INFO)
    ################################ read config , set log file ################################
    # accelerator = Accelerator()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    max_seq_len = args.max_train_length
    bs = args.bs
    lr = args.lr
    ep = args.epochs

    if (args.model == "r-f" or args.model == "f-f"):
        assert args.layers is not None
        assert args.max_train_length % args.segments == 0 
    assert args.hidden_size % args.num_attention_heads == 0
    if args.aml:
        root_dir = '/mnt/output/00ad/'
    elif args.azure:
        root_dir = '/datablob/v-zhuoli2/com_exps/'
    else:
        root_dir = '../com_exps/'
    # root_dir = '../com_exps/'
    run_dir = root_dir + args.model+ '/' + datetime.datetime.now().strftime("%Y%m%d")
    now = datetime.datetime.now().strftime("%H%M%S")
    log_name = "ads_" + \
                (("ap_" if args.ap else "cls_") if (args.model == "rob" or args.model == "fast") else "" )+\
                (("maxp_" if args.maxp else "avgp_") if (args.model == "r-f" or args.model == "f-f") else "" )+\
                ((args.layers + "_") if (args.model == "r-f" or args.model == "f-f") else "" )+\
                str(args.max_train_length) + "_"  + args.model + "_"  +\
                str(args.seed) + "_" +\
                str(args.lr) + "_" +\
                str(args.bs) + "_ep" +\
                str(args.epochs) + "_" +\
                (str(args.segments) + "_" if args.model == "r-f" or args.model == "f-f" else "") +\
                now
    os.makedirs(f'{run_dir}/{log_name}', exist_ok=True)
    
    
    handler = logging.FileHandler(f'{run_dir}/{log_name}/log.txt')
    print("log file: {}".format(f'{run_dir}/{log_name}/log.txt'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if torch.distributed.get_rank() == 0:
        logger.info(args)

    

    init_dl_program(list(range(torch.cuda.device_count())), seed=args.seed, max_threads=args.max_threads)
    # device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    ################################ text process ################################
    
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', model_max_len=60000)
    tokenizer.model_max_length = 60000
    tokenizer.init_kwargs['model_max_length'] = 60000
    mask_id = tokenizer('<mask>')['input_ids'][1]


    test_data_path = '../roberta_datasets/ads/QL.label.test.tsv'
    # test_data_path ='/datablob/v-zhuoli2/QL.label.test.tsv'
    test_data = pd.read_csv(test_data_path, delimiter="\t", names=['id', 'label','query', 'doc', 'taskid'], error_bad_lines=False)
    test_data.drop(columns = ['id','taskid'],inplace = True)
    test_data = test_data[:800]
    test_dataset = SuperviseSampleDataset(test_data, tokenizer, args.model, max_seq_len,segments=args.segments)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, collate_fn=supervise_sample_collate_fn, drop_last=True)
    label_num = len(test_data['label'].unique())
    
    if torch.distributed.get_rank() == 0:  
        print("Device: {}".format(device))
        
        print("Test data num: {}".format(len(test_data)))
        logger.info("Device: {}".format(device))

        logger.info("Test data num: {}".format(len(test_data)))
        
        
    ################################ build model ################################
    if args.model == 'r-f' or args.model == 'f-f':
        config = RobertaConfig.from_pretrained('roberta-base')
        config.num_labels = label_num
        config.max_position_embeddings = args.max_train_length
        config.model_type = 'fast-fast' if args.model == 'f-f' else 'roberta-fast'
        config.pooler_type = 'weightpooler'
        config.segments = args.segments
        config.layers = args.layers

        model = CombinedRoberta(config, segments=args.segments, finetune=args.finetune, use_ap = args.ap, use_maxp = args.maxp, mask_id = mask_id)
        
        if args.model == 'r-f':
            Roberta_base_model = RobertaModel.from_pretrained("roberta-base")
            oldModuleList = Roberta_base_model.encoder.layer
            oldEmbeddings = Roberta_base_model.embeddings
            # = model.model.teacher_model.encoder.layer
            newModuleList = nn.ModuleList()

            # Now iterate over all layers, only keepign only the relevant layers.
            for i in range(0, int(args.layers.split('-')[0])):
                newModuleList.append(oldModuleList[i])
            model.model.teacher_model.encoder.layer = newModuleList
            model.model.teacher_model.embeddings = oldEmbeddings # default segment length short than 514
        
    else:
        config = RobertaConfig.from_pretrained('roberta-base')
        config.num_labels = label_num

 
        if args.model == 'fast':
            config.max_position_embeddings = args.max_train_length
            config.model_type = 'fast'
            config.pooler_type = 'weightpooler'
            if args.drop:
                config.hidden_dropout_prob = args.drop

        model = Roberta(config, finetune=args.finetune, use_ap = args.ap, mask_id = mask_id)
        if args.model == 'rob':
            model.roberta_model.roberta = RobertaModel.from_pretrained("roberta-base")

            if args.max_train_length > model.roberta_model.roberta.embeddings.position_embeddings.weight.shape[0]:
                max_pos = args.max_train_length
                config = model.roberta_model.config

                # extend position embeddings

                current_max_pos, embed_size = model.roberta_model.roberta.embeddings.position_embeddings.weight.shape
                # max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
                config.max_position_embeddings = max_pos
                assert max_pos > current_max_pos
                # allocate a larger position embedding matrix
                new_pos_embed = model.roberta_model.roberta.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
                # copy position embeddings over and over to initialize the new position embeddings

                # k = 2
                k = 0
                step = current_max_pos - 2
                # step = current_max_pos
                while k < max_pos - 1:
                    new_pos_embed[k:(k + step)] = model.roberta_model.roberta.embeddings.position_embeddings.weight[:-2]
                    # new_pos_embed[k:(k + step)] = model.roberta_model.roberta.embeddings.position_embeddings.weight[:]
                    k += step
                new_pos_embed.to(device)
                model.roberta_model.roberta.embeddings.position_embeddings = nn.Embedding(max_pos, embed_size)
                model.roberta_model.roberta.embeddings.position_embeddings.weight.data = new_pos_embed

    if torch.distributed.get_rank() == 0:  
        logger.info(config)

    
    total = sum([param.nelement() for param in model.parameters()])
    
    if torch.distributed.get_rank() == 0:  
        print("Number of parameter: %.2fM" % (total/1e6))
        logger.info("Params: %.2fM" % (total/1e6))

    optimizer = optim.Adam([ {'params': model.parameters(), 'lr': lr}])

    model.to(device)


    model, optimizer, test_loader = accelerator.prepare(model, optimizer, test_loader)

    model.eval()

    

    t_inf = 0
    
        
    ##############################################valid eval
    if torch.distributed.get_rank() == 0:
        print("*********begin eval********")
        logger.info("*********begin eval********")

    test_latency = tester(model, test_loader, accelerator, logger,0, wandb=None, mode = "test")

    t_inf += test_latency


    
    if torch.distributed.get_rank() == 0:

        logger.info("Total inference time, {} seconds".format(datetime.timedelta(seconds=t_inf)))
        print("Total inference time, {} seconds".format(datetime.timedelta(seconds=t_inf)))
       
        logger.info("save dir: {}".format(f'{run_dir}/{log_name}'))
        print("log file: {}".format(f'{run_dir}/{log_name}/log.txt'))