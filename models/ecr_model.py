from pkgutil import get_data
import torch
import torch.nn as nn
import gsl
from utils.name2object import name2gsl, name2init
from initializer import *


class ECRModel(nn.Module):

    def __init__(self, args,  train_dataset, tokenizer, plm_model, schema_list, adj):
        super(ECRModel, self).__init__()

        # bert
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.bert_encoder = plm_model
        self.doc_schema = schema_list[0]
        self.event_schema = schema_list[1]
        self.entity_schema = schema_list[2]

        self.gsl = getattr(gsl, name2gsl[args.gsl])(args.rank, args.rank, args.rank, args.dropout)
        self.gsl_name = args.gsl_name
        self.adj = adj  # norm
        self.n_nodes = args.n_nodes

    def forward(self):
        # features: (feat_dim, n_nodes)
        
        features = torch.tensor() 
        # torch stack
        for _, sent in enumerate(self.train_dataset):
            encoder_output = self.bert_encoder(sent['input_ids'])
            encoder_hidden_state = encoder_output['last_hidden_state']  # (feat_dim, n_tokens)
            # []
            mask = torch.tensor()
            sent_mask = torch.tensor(sent['input_mask'])

            for _, event in enumerate(sent['output_event']):
                this_mask = torch.mean(sent_mask[event['tokens_number']].sum(dim=0), keepdim=True)
                mask = torch.cat((mask, this_mask), dim=0)  # dim=0 1?
            for _,entity in enumerate(sent['output_entity']):
                this_mask = torch.mean(sent_mask[entity['tokens_number']].sum(dim=0), keepdim=True)
                mask = torch.cat((mask, this_mask), dim=0)
            features = torch.cat((features, mask * encoder_hidden_state), dim=0)    #dim=0 1?encoder_hidden_state * input_mask = 所求表征

        return self.gsl(features, self.adj)  # gae, only encoder
    