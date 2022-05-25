from __future__ import absolute_import
import os
import sys
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import HyperCode
import graphzoo as gz
from graphzoo.config import parser as graphparser
from tqdm import tqdm, trange
from bleu import _bleu
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer, T5Config)
from transformers import T5EncoderModel

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript,DFG_csharp
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index, 
                   detokenize_code)
from tree_sitter import Language, Parser
sys.path.append('CodeBLEU')
from calc_code_bleu import calc_code_bleu
keywords_dir = 'CodeBLEU/keywords'

logger = logging.getLogger(__name__)

class InputFeatures(object):
    def __init__(self,
                 example_index,
                 code_tokens_ids,
                 desc_tokens_ids,
                 ast_node_types, 
                 ast_adj,
                 graph_feature
    ):
        self.example_index = example_index
        self.code_tokens_ids = code_tokens_ids
        self.desc_tokens_ids = desc_tokens_ids
        self.ast_node_types = ast_node_types
        self.ast_adj = ast_adj
        self.graph_feature = graph_feature
        
def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

       
def read_preprocessed():
    features = pickle.load(open('features_pt_java.pkl', 'rb'))
    max_code_tokens = 260
    max_desc_tokens = 260
    train_features, valid_features, test_features = features['train'], features['valid'], features['test']
    features = train_features+valid_features+test_features
    
    max_num_of_nodes = max([len(l.graph_feature) for l in features])
            
    for eg in features:
        eg.code_tokens_ids = eg.code_tokens_ids[:max_code_tokens] # is a list, not a numpy tensor
        eg.desc_tokens_ids = eg.code_tokens_ids[:max_desc_tokens] # is a list, not a numpy tensor
        adj= np.zeros((max_num_of_nodes,max_num_of_nodes))
        for nodes in list(eg.ast_adj.keys()):
            for neighbour in eg.ast_adj[nodes]:
                adj[nodes][neighbour] = 1
                adj[neighbour][nodes] = 1
        adj = normalize(adj + np.eye(adj.shape[0]))
        eg.ast_adj = adj
        for _ in range(len(eg.graph_feature),max_num_of_nodes):
            eg.graph_feature.append([0] * len(eg.graph_feature[0]))
        eg.graph_feature = normalize(np.array(eg.graph_feature))
        
    return train_features, valid_features, test_features

class TextDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples 
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        return (self.examples[item].code_tokens_ids,
                self.examples[item].desc_tokens_ids,
                self.examples[item].ast_adj,
                self.examples[item].graph_feature
               )
    
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
        
def main():
    parser = argparse.ArgumentParser()

    ## Other parameters
    parser.add_argument("--output_dir", default="saved_models/pretrain/", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument("--train_batch_size", default=5, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=5, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--test_batch_size", default=5, type=int,
                        help="Batch size per GPU/CPU for testing.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0005, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--validate_every", default=5, type=int)
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print ('**** Device *****', args.device)
    
    # Set seed
    set_seed(args.seed)
    
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
        
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    lm_model = T5EncoderModel.from_pretrained('Salesforce/codet5-base')
    
    params = graphparser.parse_args(args=[])
    params.dim=128
    params.feat_dim = 151
    params.n_nodes = 1206
    params.device=device
    params.model = 'GAT'
    params.manifold = 'Euclidean'
    params.cuda = 0
    params.num_layers = 3
    hgcn_model= gz.models.BaseModel(params).double()
        
    train_features, valid_features, test_features = read_preprocessed()
    
    model=HyperCode(lm_model=lm_model,hgcn_model = hgcn_model, device = device)

    model.to(device)
        
    def collate_batch(batch):
        code_ids_list, desc_ids_list, adj_list, feature_list = [], [], [], []
        for (code_tokens_ids, desc_tokens_ids, ast_adj, graph_feature) in batch:
            code_ids_list.append(code_tokens_ids)
            desc_ids_list.append(desc_tokens_ids)
            adj_list.append(ast_adj)
            feature_list.append(graph_feature)
            
        max_code_len_batch = max([len(l) for l in code_ids_list])
        max_desc_len_batch = max([len(l) for l in desc_ids_list])
        
        
        code_attention_mask = []
        desc_attention_mask = []
        
        for i in range(len(code_ids_list)):
            pad_len = max_code_len_batch-len(code_ids_list[i])
            code_attention_mask.append( [1]*(1+len(code_ids_list[i])) + [0]*pad_len )
            code_ids_list[i] = [tokenizer.cls_token_id] + code_ids_list[i] + \
                                    [tokenizer.pad_token_id]*pad_len
            
            pad_len = max_desc_len_batch-len(desc_ids_list[i])
            desc_attention_mask.append( [1]*(1+len(desc_ids_list[i])) + [0]*pad_len )
            desc_ids_list[i] = [tokenizer.cls_token_id] + desc_ids_list[i] + \
                                    [tokenizer.pad_token_id]*pad_len
        return torch.tensor(code_ids_list).long(), \
                torch.tensor(desc_ids_list).long(), \
                torch.tensor(code_attention_mask).int(), \
                torch.tensor(desc_attention_mask).int(), \
                torch.tensor(np.array(adj_list)).double(), \
                torch.tensor(np.array(feature_list)).double()
        

    # Prepare training data loader
    train_data = TextDataset(train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, \
                                      batch_size=args.train_batch_size, num_workers=4, collate_fn=collate_batch)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=500)
    
    #Start training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num epoch = %d", args.num_train_epochs)
        
    model.train()
    dev_dataset={}
    
    best_loss = np.inf
    loss = nn.CrossEntropyLoss()
    
    for epoch in range(args.num_train_epochs):
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        cum_loss, acc, f1, prec, rec = 0,0,0,0,0
        for batch in bar:
            batch = tuple(t.to(device) for t in batch)
            
            code_ids, desc_ids, code_attention_mask, desc_attention_mask, adj_list, feature_list  = batch                            
            code_embeds, desc_embeds =  model(code_ids, desc_ids,code_attention_mask, desc_attention_mask, adj_list, feature_list)
            
            scores = torch.matmul(code_embeds, torch.transpose(desc_embeds,0,1)).softmax(dim=1)
            target = torch.eye(scores.size(dim=1)).to(device) 

            output = loss(scores, target)
            
            _, predicted = torch.max(scores.data, 1)
            _, truth = torch.max(target.data,1)
            
            cum_loss += output.item()
            f1 += f1_score(predicted.clone().cpu().numpy(),truth.clone().cpu().numpy(),average = 'micro')
            acc += accuracy_score(predicted.clone().cpu().numpy(),truth.clone().cpu().numpy())                
            prec += precision_score(predicted.clone().cpu().numpy(),truth.clone().cpu().numpy(),average = 'micro')                
            rec += recall_score(predicted.clone().cpu().numpy(),truth.clone().cpu().numpy(),average = 'micro')                

            #Update parameters
            output.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
          
        nb_tr_steps = len(train_dataloader)
        avg_loss=round(cum_loss/nb_tr_steps,4)
        avg_acc=round(acc/nb_tr_steps,4)
        avg_f1=round(f1/nb_tr_steps,4)
        avg_prec=round(prec/nb_tr_steps,4)
        avg_rec=round(rec/nb_tr_steps,4)
        
        print('epoch '+str(epoch) +' loss '+str(avg_loss) +' acc '+str(avg_acc)+' f1 '+str(avg_f1)+' precision '+str(avg_prec)+' recall '+str(avg_rec))

        if ((epoch+1)%args.validate_every==0):
            #Eval model with dev dataset

            eval_data = TextDataset(valid_features)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, \
                                          batch_size=args.eval_batch_size, num_workers=4, collate_fn=collate_batch)

            logger.info("\n***** Running evaluation *****")

            #Start Evaling model
            model.eval()
            eval_loss = 0
            eval_acc = 0
            eval_f1 = 0
            eval_prec = 0
            eval_rec = 0

            for batch in tqdm(eval_dataloader):
                batch = tuple(t.to(device) for t in batch)
                code_ids, desc_ids, code_attention_mask, desc_attention_mask, adj_list, feature_list = batch

                with torch.no_grad():
                    code_embeds, desc_embeds =  model(code_ids, desc_ids,code_attention_mask, desc_attention_mask, adj_list, feature_list)

                    scores = torch.matmul(code_embeds, torch.transpose(desc_embeds,0,1)).softmax(dim=1)
                    target = torch.eye(scores.size(dim=1)).to(device) 

                    output = loss(scores, target)

                    _, predicted = torch.max(scores.data, 1)
                    _, truth = torch.max(target.data,1)
                    
                    eval_loss += output.item()
                    eval_f1 += f1_score(predicted.clone().cpu().numpy(),truth.clone().cpu().numpy(),average = 'micro')
                    eval_acc += accuracy_score(predicted.clone().cpu().numpy(),truth.clone().cpu().numpy())
                    eval_prec += precision_score(predicted.clone().cpu().numpy(),truth.clone().cpu().numpy(),average = 'micro')
                    eval_rec += recall_score(predicted.clone().cpu().numpy(),truth.clone().cpu().numpy(),average = 'micro')
                    

            nb_eval_steps = len(eval_dataloader)

            #Print loss of dev dataset 
            result = {'eval_loss': round(eval_loss/nb_eval_steps,5),
                      'eval_acc': round(eval_acc/nb_eval_steps,5),
                      'eval_f1': round(eval_f1/nb_eval_steps,5),
                      'eval_prec': round(eval_prec/nb_eval_steps,5),
                      'eval_rec': round(eval_rec/nb_eval_steps,5),
                      'global_step': epoch+1}

            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
            logger.info("  "+"*"*20)   

            #save last checkpoint
            last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
            if not os.path.exists(last_output_dir):
                os.makedirs(last_output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model 
            output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)  

            # save best loss and ppl
            if eval_loss/nb_eval_steps<best_loss:
                logger.info("  Best loss:%s",round(eval_loss/nb_eval_steps,5))
                logger.info("  "+"*"*20)
                best_loss=eval_loss/nb_eval_steps
                output_dir = os.path.join(args.output_dir, 'checkpoint-best-loss')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)  
    
    test_data = TextDataset(test_features)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, \
                                          batch_size=args.test_batch_size, num_workers=4, collate_fn=collate_batch)

    logger.info("\n***** Running Testing *****")

    #Start Testing model
    model.load_state_dict(torch.load(output_model_file,map_location=lambda storage, loc: storage)) 
    model.eval()

    test_loss = 0
    test_acc = 0
    test_f1 = 0
    test_prec = 0
    test_rec = 0

    for batch in tqdm(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        code_ids, desc_ids, code_attention_mask, desc_attention_mask, adj_list, feature_list = batch

        with torch.no_grad():
            code_embeds, desc_embeds =  model(code_ids, desc_ids,code_attention_mask, desc_attention_mask, adj_list, feature_list)

            scores = torch.matmul(code_embeds, torch.transpose(desc_embeds,0,1)).softmax(dim=1)
            target = torch.eye(scores.size(dim=1)).to(device) 

            output = loss(scores, target)

            _, predicted = torch.max(scores.data, 1)
            _, truth = torch.max(target.data,1)

            test_loss += output.item()
            test_f1 += f1_score(predicted.clone().cpu().numpy(),truth.clone().cpu().numpy(),average = 'micro')
            test_acc += accuracy_score(predicted.clone().cpu().numpy(),truth.clone().cpu().numpy())
            test_prec += precision_score(predicted.clone().cpu().numpy(),truth.clone().cpu().numpy(),average = 'micro')
            test_rec += recall_score(predicted.clone().cpu().numpy(),truth.clone().cpu().numpy(),average = 'micro')


    nb_test_steps = len(test_dataloader)

    #Print loss of dev dataset 
    result = {'test_loss': round(test_loss/nb_test_steps,5),
              'test_acc': round(test_acc/nb_test_steps,5),
              'test_f1': round(test_f1/nb_test_steps,5),
              'test_prec': round(test_prec/nb_test_steps,5),
              'test_rec': round(test_rec/nb_test_steps,5)}
    
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    logger.info("  "+"*"*20)  

if __name__ == "__main__":
    main()