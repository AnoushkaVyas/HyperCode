# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
from torch.autograd import Variable
import copy


class HyperCode(nn.Module):

    def __init__(self, lm_model= None, hgcn_model=None, beam_size=None, max_length=None, sos_id=None, eos_id=None, pad_id=None, unk_id=None,max_depth=None, num_node_types=None, device = None):
        super(HyperCode, self).__init__()
        
        self.lm_model = lm_model 
        self.hgcn_model = hgcn_model
        self.device = device

        
    def forward(self, code_ids, desc_ids, code_attention_mask, desc_attention_mask, adj_list, feature_list): 
        # code_ids : b, L1
        # desc_ids : b, L2
        # code_attention_mask : b, L1 binary, 1 for not padded, 0 for padded
        
        
        code_outputs = self.lm_model(input_ids=code_ids, attention_mask=code_attention_mask, return_dict=True)
        code_outputs = code_outputs.last_hidden_state # b, L1, 768
        code_embeds = code_outputs[:,0,:] # b,768     CLS token embedding at last encoder layer
                
        desc_outputs = self.lm_model(input_ids=desc_ids, attention_mask=desc_attention_mask, return_dict=True)
        desc_outputs = desc_outputs.last_hidden_state # b, L1, 768
        desc_embeds = desc_outputs[:,0,:] # b,768     CLS token embedding at last encoder layer  
        
        graph_embedding = torch.zeros(code_embeds.size(0),1206,128)
        for i in range(adj_list.shape[0]):
            graph_embedding[i] = self.hgcn_model.encode(feature_list[i], adj_list[i])

        graph_embedding = graph_embedding.to(self.device)
        code_embeds = torch.cat((code_embeds,graph_embedding.flatten(start_dim = 1)), 1)
        desc_embeds = torch.cat((desc_embeds,graph_embedding.flatten(start_dim = 1)), 1)
        
        return code_embeds,desc_embeds
      
