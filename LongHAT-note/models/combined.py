from distutils.command.config import config

from matplotlib.pyplot import axes
import torch
import torch.nn as nn
import pdb
import logging
import time
import random
import numpy as np
from models.bert import BertSelfOutput, BertIntermediate, BertOutput
from models.roberta import RobertaForSequenceClassification, RobertaForMaskedLM, RobertaConfig
from models.roberta import CombinedRobertaForSequenceClassification, CombinedRobertaForMaskedLM
import global_time
import copy
logger = logging.getLogger('__main__.model')

#############note####################
#here fastformer model is not in use

class AttentionPooling(nn.Module):
    def __init__(self, config):
        self.config = config
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.att_fc2 = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
                
    def forward(self, x, attn_mask=None):
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  
        return x

class FastSelfAttention(nn.Module):
    def __init__(self, config):
        super(FastSelfAttention, self).__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        self.attention_head_size = int(config.hidden_size /config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.input_dim= config.hidden_size
        
        self.query = nn.Linear(self.input_dim, self.all_head_size)
        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key = nn.Linear(self.input_dim, self.all_head_size)
        self.key_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
                
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask):
        # batch_size, seq_len, num_head * head_dim, batch_size, seq_len
        batch_size, seq_len, _ = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        # batch_size, num_head, seq_len
        query_for_score = self.query_att(mixed_query_layer).transpose(1, 2) / self.attention_head_size**0.5
        # add attention mask
        query_for_score += attention_mask

        # batch_size, num_head, 1, seq_len
        query_weight = self.softmax(query_for_score).unsqueeze(2)

        # batch_size, num_head, seq_len, head_dim
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # batch_size, num_head, head_dim, 1
        pooled_query = torch.matmul(query_weight, query_layer).transpose(1, 2).view(-1,1,self.num_attention_heads*self.attention_head_size)
        pooled_query_repeat= pooled_query.repeat(1, seq_len,1)
        # batch_size, num_head, seq_len, head_dim

        # batch_size, num_head, seq_len
        mixed_query_key_layer=mixed_key_layer* pooled_query_repeat
        
        query_key_score=(self.key_att(mixed_query_key_layer)/ self.attention_head_size**0.5).transpose(1, 2)
        
        # add attention mask
        query_key_score +=attention_mask

        # batch_size, num_head, 1, seq_len
        query_key_weight = self.softmax(query_key_score).unsqueeze(2)

        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = torch.matmul(query_key_weight, key_layer)

        #query = value
        weighted_value =(pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2] + (self.num_attention_heads * self.attention_head_size,))
        weighted_value = self.transform(weighted_value) + mixed_query_layer
      
        return weighted_value
    
class FastAttention(nn.Module):
    def __init__(self, config):
        super(FastAttention, self).__init__()
        self.self = FastSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class FastformerLayer(nn.Module):
    def __init__(self, config):
        super(FastformerLayer, self).__init__()
        self.attention = FastAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
class FastformerEncoder(nn.Module):
    def __init__(self, config, pooler_count=1):
        super(FastformerEncoder, self).__init__()
        self.config = config
        self.encoders = nn.ModuleList([FastformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # support multiple different poolers with shared bert encoder.
        self.poolers = nn.ModuleList()
        if config.pooler_type == 'weightpooler':
            for _ in range(pooler_count):
                self.poolers.append(AttentionPooling(config))
        logging.info(f"This model has {len(self.poolers)} poolers.")

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, 
                input_embs, 
                attention_mask, 
                pooler_index=0):
        #input_embs: batch_size, seq_len, emb_dim
        #attention_mask: batch_size, seq_len, emb_dim
        
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        batch_size, seq_length, emb_dim = input_embs.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embs.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embs + position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        #print(embeddings.size())
        all_hidden_states = [embeddings]

        for i, layer_module in enumerate(self.encoders):
            # pdb.set_trace()
            layer_outputs = layer_module(all_hidden_states[-1], extended_attention_mask)
            all_hidden_states.append(layer_outputs)
        # pdb.set_trace()
        assert len(self.poolers) > pooler_index
        output = self.poolers[pooler_index](all_hidden_states[-1], attention_mask)

        return output 

class FastFormer(torch.nn.Module):
    
    def __init__(self,config,word_dict=None,num_classes=None):
        super(FastFormer, self).__init__()
        self.config = config
        self.dense_linear = nn.Linear(config.hidden_size, num_classes)
        # self.word_embedding = nn.Embedding(len(word_dict),256,padding_idx=0)
        self.word_embedding = nn.Embedding(len(word_dict),config.hidden_size,padding_idx=0)
        self.fastformer_model = FastformerEncoder(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self,input_ids,targets):
        # pdb.set_trace()
        mask=input_ids.bool().float()
        embds=self.word_embedding(input_ids)
        text_vec = self.fastformer_model(embds,mask)
        # print(text_vec.shape)
        score = self.dense_linear(text_vec)
        loss = self.criterion(score, targets) 
        return loss, score

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_list,use_ap=False, mask_id = 0):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    oringin_tokens = copy.deepcopy(tokens)
    vocab_list = np.arange(vocab_list).tolist()
    cand_indices = []
    if not use_ap:
        cand_list = np.arange(tokens.shape[1]-1) + 1
    else:
        cand_list = np.arange(tokens.shape[1])

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(tokens.shape[1] * masked_lm_prob))))

    for bs_index in range(tokens.shape[0]):

        cand_indices = cand_list.tolist()
        random.shuffle(cand_indices)
        mask_indices = sorted(random.sample(cand_indices, num_to_mask))

        
        for index in mask_indices:
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = -1
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = tokens[bs_index][index]
                # 10% of the time, replace with random word
                else:
                    masked_token = random.choice(vocab_list)
            # masked_token_labels.append(tokens[bs_index][index])
            # Once we've saved the true label for that token, we can overwrite it with the masked version
            tokens[bs_index][index] = masked_token
            if masked_token == -1:
                oringin_tokens[bs_index][index] = mask_id
            else:
                oringin_tokens[bs_index][index] = masked_token

    return oringin_tokens, tokens

class Roberta(torch.nn.Module):
    
    def __init__(self,config, finetune = False, use_ap =False, mask_id = 0):
        super(Roberta, self).__init__()
        self.config = config
        self.roberta_model = RobertaForSequenceClassification(config, use_ap=use_ap) if finetune else RobertaForMaskedLM(config)
        self.criterion = nn.CrossEntropyLoss()  if finetune else None
        self.apply(self.init_weights)
        self.finetune = finetune
        self.use_ap = use_ap
        self.mask_id = mask_id
        
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self,input_ids,targets=None, token_ids=None):

        x = input_ids
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if self.finetune:
            
            mask = (x!=1).bool().float()
            score = self.roberta_model.forward(x, attention_mask=mask,position_ids=position_ids)[0]
            loss = self.criterion(score, targets) 
        else:
            (x,
            masked_lm_labels) = create_masked_lm_predictions(
                    x, masked_lm_prob = 0.15, 
                    max_predictions_per_seq=self.config.max_position_embeddings, 
                    vocab_list=self.config.vocab_size,
                    use_ap = self.use_ap,
                    mask_id=self.mask_id)
           
            mask = (x!=1).bool().float()
           
            loss = self.roberta_model.forward(x, attention_mask=mask,masked_lm_labels=masked_lm_labels,position_ids=position_ids)[0]
            score = None
        return loss, score
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.roberta_model.roberta.state_dict(), fn)
    
    def load(self, fn,device = 'cuda'):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''

        import os
        print(os.path.exists(fn))
        state_dict = torch.load(fn, map_location=device)
        self.roberta_model.roberta.load_state_dict(state_dict)



class CombinedRoberta(torch.nn.Module):
    
    def __init__(self,config, segments = 4, finetune=False, use_ap =False, use_maxp = False, mask_id = 0 ):
        super(CombinedRoberta, self).__init__()
        self.config = config
      
        self.model = CombinedRobertaForSequenceClassification(self.config,
                    segments, use_ap =use_ap, use_maxp = use_maxp)  if finetune else CombinedRobertaForMaskedLM(config)
        self.criterion = nn.CrossEntropyLoss() if finetune else None
        self.segments = segments
        self.apply(self.init_weights)
        self.finetune = finetune
        self.use_ap = use_ap        
        self.mask_id = mask_id
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self,input_ids,targets=None, token_ids=None):
      
        t = time.time()
        if input_ids.shape[0] == 0:
            pdb.set_trace()

        ##########################Segmentation######################
        m, n = input_ids.shape
        input_segs  = input_ids.reshape(m * self.segments, n // self.segments).cpu()
        
        # zerorows -- we need to prune 
        zerorows = np.where((input_segs==1).all(axis=1))[0]
        
        # userows -- we need to forward to encoder
        userows = np.where(~(input_segs==1).all(axis=1))[0]
        use_input_ids = input_segs[userows,:]
        cnt = 0
        i = 0
        cur_end = -1
        use_input_flag = 0

        while i < len(zerorows):
            before_seg_cnts = zerorows[i] // self.segments
            res_seg_cnts = (zerorows[i] - cur_end -1) // self.segments

            if cnt == 0:
                
                before = np.repeat( self.segments, before_seg_cnts, axis = 0)
                cnt += 1
                cur_end = before_seg_cnts * self.segments + self.segments-1
                cur_seg_cnts = self.segments
                while i < len(zerorows)  and zerorows[i] <= cur_end :
                    cur_seg_cnts = cur_seg_cnts - 1
                    i += 1
                use_input_flag = np.append(before, cur_seg_cnts)
            else:
                before = np.append(use_input_flag, np.repeat( self.segments, res_seg_cnts, axis = 0))
                cur_end = before_seg_cnts * self.segments + self.segments-1
                cur_seg_cnts = self.segments

                while i < len(zerorows)  and zerorows[i] <= cur_end :
                    cur_seg_cnts = cur_seg_cnts - 1
                    i += 1
                use_input_flag = np.append(before, cur_seg_cnts)
            

        if cur_end < len(input_segs)-1:
 
            res_seg_cnts = (len(input_segs) - cur_end -1) // self.segments
            if cur_end == -1:
                use_input_flag = np.repeat( self.segments, res_seg_cnts, axis = 0)
            else:
                use_input_flag = np.append(use_input_flag,np.repeat( self.segments, res_seg_cnts, axis = 0)) 
     
        if  type(use_input_flag) == int:
            pdb.set_trace()     
        # print('Segments flag validation: ', np.sum(use_input_flag) == len(use_input_ids))
        if not  np.sum(use_input_flag) == len(use_input_ids):
            pdb.set_trace()

        x = torch.LongTensor(use_input_ids).to(input_ids.device)

        if self.finetune:
            
            mask = (x!=1).bool().float()
            score = self.model.forward(x, attention_mask=mask, seg_flags = use_input_flag)[0]
            loss = self.criterion(score, targets) 

        else:
            
            (x,
            masked_lm_labels) = create_masked_lm_predictions(
                    x, masked_lm_prob = 0.15, 
                    max_predictions_per_seq=self.config.max_position_embeddings // self.segments, 
                    vocab_list=self.config.vocab_size,
                    use_ap = self.use_ap,
                    mask_id=self.mask_id)

            mask = (x!=1).bool().float()

            loss = self.model.forward(x, attention_mask=mask,masked_lm_labels=masked_lm_labels, seg_flags = use_input_flag)[0]
            score = None
  
        return loss, score
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        # pdb.set_trace()
        model_dict = {}
        model_dict['teacher'] = self.model.teacher_model.state_dict()
        model_dict['combined'] = self.model.combined_model.state_dict()
        torch.save(model_dict, fn)
    
    def load(self, fn,device = 'cuda'):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''

        import os
        print(os.path.exists(fn))
        state_dict = torch.load(fn, map_location=device)
        self.model.teacher_model.load_state_dict(state_dict['teacher'])
        self.model.combined_model.load_state_dict(state_dict['combined'])

def acc(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score

def Evaluate(y_true,y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average='macro'),
        "micro_precision": precision_score(y_true, y_pred, average='micro'),
        "macro_recall": recall_score(y_true, y_pred, average='macro'),
        "micro_recall": recall_score(y_true, y_pred, average='micro'),
        "macro_f": f1_score(y_true, y_pred, average='macro'),
        "micro_f": f1_score(y_true, y_pred, average='micro')
    }
 