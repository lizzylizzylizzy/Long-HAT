######################
# written by zhuo li
# intern project
# 2022.9.24
######################
import numpy as np
import pandas as pd
import os
import logging

import argparse
import datetime
import pdb
import json
import torch
# import deepspeed
import torch.nn as nn

import time
from statistics import mean
from tqdm.auto import tqdm
from collections import Counter
from sklearn.metrics import *
from pytorch_transformers import RobertaConfig
from transformers import RobertaTokenizer, RobertaModel, get_scheduler, AdamW
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from models.combined import Roberta, CombinedRoberta, Evaluate
from utils import init_dl_program, supervise_sample_collate_fn, SuperviseSampleDataset

# import matplotlib.pyplot as plt

# import global_time 
# import copy

#########note##############
# if torch.distributed.get_rank() == 0:
# mainly for unrepeated log 
# you can use accelerator.log for simplification
 
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

def get_optimizer_params(model, type='rob', plus =1, learning_rate=1e-5):
    # Differential learning rate and weight decay
    param_optimizer = list(model.named_parameters())
    weight_decay = 0.01
    no_decay = ['bias', 'gamma', 'beta']
    if plus == 1:
        optimizer_parameters = filter(lambda x: x.requires_grad, model.parameters())
    elif (type == 'rob' or type == 'fast') :
        optimizer_parameters = [
            {'params': [p for n, p in model.roberta_model.roberta.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': weight_decay},
            {'params': [p for n, p in model.roberta_model.roberta.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0},
            {'params': [p for n, p in model.named_parameters() if 'classifier' in n],
            'lr': learning_rate*plus,
            'weight_decay_rate':weight_decay}
        ]
    elif type == 'r-f':
        optimizer_parameters = [
            {'params': [p for n, p in model.model.teacher_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': weight_decay},
            {'params': [p for n, p in model.model.teacher_model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0},
            {'params': [p for n, p in model.model.named_parameters() if "combined_model" in n],
             'lr': learning_rate*plus,
             'weight_decay_rate':weight_decay}
        ]

    return optimizer_parameters

def tester(model, loader, accelerator, logger, epoch, wandb=None, mode=None):
    # Wrap up test process, applicable to valid and test
    allpred_test=[]
    alltrue_test=[]
    tester_loss = 0.0
    t_inf_start = time.time()
    for iter_index , (log_ids, targets) in enumerate(loader, 1):   
        with accelerator.accumulate(model):

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
    all_pred_test_well = 1-allpred_test_np[:,0]
    
    y_true = np.array(alltrue_test)
    y_true[y_true>1] = 1
    
    test_auc_score = roc_auc_score(y_true, all_pred_test_well)

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
    parser.add_argument('--save_interval', type=int, default=100, help='Save checkpoint interval(default to 100), current code not in use')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--lr', type=float, default=0.00001, help='The learning rate (defaults to 0.00001)')
    parser.add_argument('--drop', type=float, default=None, help='The drop rate (defaults to 0.1)')
    parser.add_argument('--aml', action="store_true", help='Whether on aml, used to load data')
    parser.add_argument('--bs', type=int, default=32, help='The batch size (defaults to 32)')
    parser.add_argument('--max_train_length', type=int, default=256, help='max sequence length, this code will cut the excessive part directly (defaults to 256)')
    parser.add_argument('--segments', type=int, default=4, help='The segments (defaults to 4)')
    parser.add_argument('--epochs', type=int, default=3, help='The number of epochs')
    parser.add_argument('--num_hidden_layers', type=int, default=4, help='The hidden layer (defaults to 4)')
    parser.add_argument('--layers', type=str, default=None, help='Layers config for Long-HAT, format a-b, a is model1 layer, b is model2 layer')
    parser.add_argument('--num_attention_heads', type=int, default=16, help='The attention heads (defaults to 16)')
    parser.add_argument('--hidden_size', type=int, default=256, help='The hidden size (defaults to 256)')
    parser.add_argument('--warm_step', type=int, default=400, help='Warm up step for lr scheduler')
    parser.add_argument('--intermediate_size', type=int, default=1024, help='The intermediate size (defaults to 1024)')
    parser.add_argument('--grad_step', type=int, default=1, help='The gradient accumulation steps (defaults to 2)')
    parser.add_argument('--finetune', action="store_true" , help='Finetune stage or Pretrain, currently only fintune in use')
    parser.add_argument('--ap', action="store_true" , help='attention pooler')
    parser.add_argument('--maxp', action="store_true" , help='use maxpooling to extract embeddings over segment axis, if not set, we use avg. pooling')
    parser.add_argument('--azure', action="store_true" , help='set if code running on azure, used to load data')
    parser.add_argument('--plus', type=float, default=1, help='for task-specific scheduler')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    # parser = deepspeed.add_config_arguments(parser)
    
    ################################ read config , set log file ################################
    # accelerator = Accelerator()
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    gradient_accumulation_steps = args.grad_step
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs],gradient_accumulation_steps=gradient_accumulation_steps)
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

    run_dir = root_dir + args.model+ '/' + datetime.datetime.now().strftime("%Y%m%d")
    now = datetime.datetime.now().strftime("%H%M%S")
    log_name = "ads_" + \
                (("ap_" if args.ap else "cls_") if (args.model == "rob" or args.model == "fast") else "" )+\
                (("maxp_" if args.maxp else "avgp_") if (args.model == "r-f" or args.model == "f-f") else "" )+\
                ((args.layers + "_") if (args.model == "r-f" or args.model == "f-f") else "" )+\
                str(args.max_train_length) + "_"  + args.model + "_"  +\
                str(args.seed) + "_" +\
                str(args.lr) + "_" +\
                str(args.plus) + "_" +\
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

    #############note##################
    #you can reference if you want to log by wandb
    
    # wandb_data_save_dir = '/data/data2/v-zhuoli2/Combined_log' 
    # if torch.distributed.get_rank() == 0:  
    #     wandb.init( settings=wandb.Settings(start_method="fork"),
    #                 project="exp" + datetime.datetime.now().strftime("%Y%m%d"), 
    #                 entity="look", 
    #                 name=f'{log_name}',
    #                 dir = root_dir)
    #     wandb.config =  {
    #     "learning_rate": args.lr,
    #     "epochs": args.epochs,
    #     "batch_size": args.bs,
    #     "model_type": args.model,
    #     "max_seq_length": args.max_train_length
    #     }
    

    init_dl_program(list(range(torch.cuda.device_count())), seed=args.seed, max_threads=args.max_threads)
    # device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    
    ################################ text process ################################

    if args.aml:
        train_data_path ='/mnt/output/aadata/QL.label.train.tsv'
    elif args.azure:
        train_data_path ='/datablob/v-zhuoli2/QL.label.train.tsv'
    else:
        train_data_path = '../roberta_datasets/ads/QL.label.test.tsv'

    train_data = pd.read_csv(train_data_path, delimiter="\t", names=['id', 'label','query', 'doc', 'taskid'], error_bad_lines=False)
    train_data.drop(columns = ['id','taskid'],inplace = True)
    label_num = len(train_data['label'].unique())
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', model_max_len=60000)
    tokenizer.model_max_length = 60000
    tokenizer.init_kwargs['model_max_length'] = 60000
    mask_id = tokenizer('<mask>')['input_ids'][1]

    train_data_ratio = 1
    if not (args.azure or args.aml):
        train_data_ratio = 0.8
    real_train_data=train_data[:int(len(train_data)*train_data_ratio)]
    
    ####### here used for valid data spiltting
    # real_train_data=train_data[:int(len(train_data)*0.8*train_data_ratio)]
    # valid_data=train_data[int(len(train_data)*0.8*train_data_ratio):int(len(train_data)*train_data_ratio)].reset_index(drop=True)

    train_dataset = SuperviseSampleDataset(real_train_data, tokenizer, args.model, max_seq_len,segments=args.segments)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, collate_fn=supervise_sample_collate_fn, drop_last=True)

    # valid_dataset = SuperviseSampleDataset(valid_data, tokenizer, args.model, max_seq_len,segments=args.segments)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.bs, shuffle=False, collate_fn=supervise_sample_collate_fn, drop_last=True)
    
    if args.finetune:
        if args.aml:
            test_data_path = '/mnt/output/aadata/QL.label.test.tsv'
            test_data = pd.read_csv(test_data_path, delimiter="\t", names=['id', 'label','query', 'doc', 'taskid'], error_bad_lines=False)
            test_data.drop(columns = ['id','taskid'],inplace = True)
            test_dataset = SuperviseSampleDataset(test_data, tokenizer, args.model, max_seq_len,segments=args.segments)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, collate_fn=supervise_sample_collate_fn, drop_last=True)
        elif args.azure:
            test_data_path = '/datablob/v-zhuoli2/QL.label.test.tsv'
            test_data = pd.read_csv(test_data_path, delimiter="\t", names=['id', 'label','query', 'doc', 'taskid'], error_bad_lines=False)
            test_data.drop(columns = ['id','taskid'],inplace = True)
            test_dataset = SuperviseSampleDataset(test_data, tokenizer, args.model, max_seq_len,segments=args.segments)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, collate_fn=supervise_sample_collate_fn, drop_last=True)
        else:
            test_data = train_data[int(len(train_data)*train_data_ratio):].reset_index(drop=True)
            test_dataset = SuperviseSampleDataset(test_data, tokenizer, args.model, max_seq_len,segments=args.segments)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, collate_fn=supervise_sample_collate_fn, drop_last=True)
    else:
        test_data=[]
        test_loader = None
    
    if torch.distributed.get_rank() == 0:  
        print("Device: {}".format(device))
        print("Train data num: {}".format(len(real_train_data)))
        # print("Valid data num: {}".format(len(valid_data)))
        print("Test data num: {}".format(len(test_data)))
        logger.info("Device: {}".format(device))
        logger.info("Train data num: {}".format(len(real_train_data)))
        # logger.info("Valid data num: {}".format(len(valid_data)))
        logger.info("Test data num: {}".format(len(test_data)))
        
        
    ################################ build model ################################
    if args.model == 'r-f' or args.model == 'f-f' or args.model == 'r-r':
        config = RobertaConfig.from_pretrained('roberta-base')
        config.num_labels = label_num
        config.max_position_embeddings = args.max_train_length
        if args.model == 'f-f': 
            config.model_type = 'fast-fast' 
        elif args.model == 'r-f': 
            config.model_type ='roberta-fast'
        elif args.model == 'r-r': 
            config.model_type ='roberta-roberta'
        config.pooler_type = 'weightpooler'
        config.segments = args.segments
        config.layers = args.layers

        model = CombinedRoberta(config, segments=args.segments, finetune=args.finetune, use_ap = args.ap, use_maxp = args.maxp, mask_id = mask_id)
        
        if args.model == 'r-f' or args.model == 'r-r':
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
                    # new_pos_embed[k:(k + step)] = model.roberta_model.roberta.embeddings.position_embeddings.weight[2:]
                    k += step
                new_pos_embed.to(device)
                model.roberta_model.roberta.embeddings.position_embeddings = nn.Embedding(max_pos, embed_size)
                model.roberta_model.roberta.embeddings.position_embeddings.weight.data = new_pos_embed
                # model.roberta_model.roberta.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)


    if torch.distributed.get_rank() == 0:  
        logger.info(config)


    total = sum([param.nelement() for param in model.parameters()])
    
    if torch.distributed.get_rank() == 0:  
        print("Number of parameter: %.2fM" % (total/1e6))
        logger.info("Params: %.2fM" % (total/1e6))
    
    parameters = get_optimizer_params(model, type = args.model, plus = args.plus, learning_rate=lr)
    kwargs = {
        'betas': (0.9, 0.999),
        'eps': 1e-08
    }
    optimizer = AdamW(parameters, lr=lr, **kwargs)
    num_training_steps = ep * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=args.warm_step, num_training_steps=num_training_steps
    )

    model.to(device)
    
   
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader,lr_scheduler)
    # valid_loader = accelerator.prepare(valid_loader)
    test_loader = accelerator.prepare(test_loader) if test_loader else None
    ################################Pretrain################################
    ########note########
    # current pretrain is not in use, may rise many bugs if you directly run in pretrain stage
    if not args.finetune:
        t_all = time.time()
        t_train = 0
        for epoch in range(ep):
            t_train_start = time.time()
            loss = 0.0
            accuary = 0.0
            for iter_index , (log_ids, _) in enumerate(train_loader, 1):
                log_ids= log_ids.to(device)
                bz_loss, _ = model(log_ids) 
                loss += bz_loss.data.float()
                unified_loss=bz_loss
                optimizer.zero_grad()
                accelerator.backward(unified_loss)
                optimizer.step()

                if iter_index % args.log_interval == 0 and torch.distributed.get_rank() == 0:
                        print( ' Iter: {}, train_loss: {:.5f}'.format(iter_index, loss.data / iter_index))
                        logger.info( ' Iter: {}, train_loss: {:.5f}'.format(iter_index, loss.data / iter_index))
                        # wandb.log({"loss": loss.data / iter_index})
                    # model.save(f'{run_dir}/{log_name}/model_{iter_index}.pkl')
            t_train += time.time() - t_train_start
            org_training = model.training
            model.eval()
            
            ##############################################valid eval

            valid_loss = 0.0 
            for iter_index , (log_ids, _) in enumerate(valid_loader, 1):   
                log_ids= log_ids.to(device)
                with torch.no_grad():
                    bz_loss2,_ = model(log_ids)
                
                loss_gatherd = accelerator.gather(bz_loss2) 
                valid_loss += loss_gatherd.sum().data.float()
            valid_loss = valid_loss.data / iter_index
            if torch.distributed.get_rank() == 0:  
                print(' Epoch: {},  valid loss: {:.5f}'.format(epoch, valid_loss.item()))
                logger.info( ' Epoch: {},  valid loss: {:.5f}'.format(epoch, valid_loss.item()))
                model.train(org_training)
        if torch.distributed.get_rank() == 0:  
            logger.info("Total pretrain time, {} seconds".format(datetime.timedelta(seconds=t_train)))
            print("Total pretrain time, {} seconds".format(datetime.timedelta(seconds=t_train)))
            print("log file: {}".format(f'{run_dir}/{log_name}/log.txt'))
    
    ################################Finetune################################
    else:
        ##use for load checkpoint
        # save_dir = "../com_exps/r-f/20220815/ads_avgp_10-2_512_r-f_42_2.5e-05_8_ep3_2_185528/cp_2_1350.pth"
        # unwrapped_model = accelerator.unwrap_model(model)
        # unwrapped_model.load_state_dict(torch.load(f'{save_dir}'))

        
        t_all = time.time()
        t_train = 0
        t_inf = 0
        lr_list = []
        # progress_bar = tqdm(range(num_training_steps))
        for epoch in range(ep):
            t_train_start = time.time()
            loss = 0.0
            accuary = 0.0
            for iter_index , (log_ids, targets) in enumerate(train_loader, 1):  
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    log_ids= log_ids.to(device)
                    targets= targets.to(device)
                    bz_loss, y_hat = model(log_ids, targets) 
                
                    loss += bz_loss.data.float()
                    unified_loss=bz_loss
                    accelerator.backward(unified_loss)
                    optimizer.step()
                    # lr_list.append(optimizer.param_groups[0]['lr'])
                    lr_scheduler.step()
                    # progress_bar.update(1)


                if iter_index % args.log_interval== 0 and torch.distributed.get_rank() == 0 :
                    print( ' Iter: {}, train_loss: {:.5f}'.format(iter_index, loss.data / iter_index))
                    logger.info( ' Iter: {}, train_loss: {:.5f}'.format(iter_index, loss.data / iter_index))
                    # wandb.log({"loss": loss.data / iter_index})
                # if iter_index % args.save_interval== 0:
            ##use to see lr process scheduled
            # plt.plot(list(range(len(train_loader))), lr_list)
            # plt.xlabel("epoch")
            # plt.ylabel("lr")
            # plt.title("CosineAnnealingLR")
            # plt.savefig('lr.png', dpi = 400,bbox_inches='tight')
            # plt.show()
            
            ##use to save checkpoints
            # accelerator.wait_for_everyone()
            # if torch.distributed.get_rank() == 0 :
            #     unwrapped_model = accelerator.unwrap_model(model)
            #     # save
            #     accelerator.save(unwrapped_model.state_dict(), f'{run_dir}/{log_name}/cp_{epoch}_{iter_index}.pth')
            #     print("################### save path: {}".format(f'{run_dir}/{log_name}/cp_{epoch}_{iter_index}.pth'))
            #     logger.info("cp save path: {}".format(f'{run_dir}/{log_name}/cp_{epoch}_{iter_index}.pth'))
            
            cur_epoch_train_time = time.time() - t_train_start
            
            if torch.distributed.get_rank() == 0 :
                logger.info("Epoch {} train time, {} seconds".format(epoch, datetime.timedelta(seconds=cur_epoch_train_time)))
                print("Epoch {} train time, {} seconds".format(epoch, datetime.timedelta(seconds=cur_epoch_train_time)))
            
            t_train += cur_epoch_train_time
            org_training = model.training
            model.eval()
            
            ##############################################valid eval
            if torch.distributed.get_rank() == 0:
                print("*********begin eval********")
                logger.info("*********begin eval********")
            # valid_latency = tester(model, valid_loader, accelerator, logger,epoch, wandb=None, mode = "dev")
            test_latency = tester(model, test_loader, accelerator, logger,epoch, wandb=None, mode = "test")

            t_inf += test_latency
            model.train(org_training)

        t_all = time.time() - t_all
        
        if torch.distributed.get_rank() == 0:
            logger.info("Total train time, {} seconds".format(datetime.timedelta(seconds=t_train)))
            print("Total train time, {} seconds".format(datetime.timedelta(seconds=t_train)))
            logger.info("Total inference time, {} seconds".format(datetime.timedelta(seconds=t_inf)))
            print("Total inference time, {} seconds".format(datetime.timedelta(seconds=t_inf)))
            logger.info("Total time, {} seconds".format(datetime.timedelta(seconds=t_all)))
            print("Total time, {} seconds".format(datetime.timedelta(seconds=t_all)))
            logger.info("save dir: {}".format(f'{run_dir}/{log_name}'))
            print("log file: {}".format(f'{run_dir}/{log_name}/log.txt'))