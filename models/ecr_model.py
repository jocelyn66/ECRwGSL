from cgi import test
from json import encoder
from turtle import shape
import torch
import torch.nn as nn
from utils.name2object import name2gsl, name2init
import models.gsl as gsls
import tqdm


class ECRModel(nn.Module):

    def __init__(self, args, tokenizer, plm_model, schema_list):
        super(ECRModel, self).__init__()

        # bert
        self.tokenizer = tokenizer
        self.bert_encoder = plm_model
        self.doc_schema = schema_list[0]
        self.event_schema = schema_list[1]
        self.entity_schema = schema_list[2]

        self.gsl = getattr(gsls, name2gsl[args.encoder])(args.feat_dim, args.hidden1, args.hidden2, args.dropout)
        self.gsl_name = args.encoder
        self.device = args.device

    def forward(self, dataset, adj):
        # features: (feat_dim, n_nodes)
        # normalized adj
        # 待优化: 高维张量
        features = []  # ts list

        # 遍历句子构造句子子图, 同时记录句子文档id
        for _, sent in enumerate(dataset):
            # print("#sent", _)
            input_ids = torch.tensor(sent['input_ids'], device=self.device).reshape(1, -1)
            encoder_output = self.bert_encoder(input_ids)
            # print("#####input_ids", input_ids.shape)
            encoder_hidden_state = encoder_output['last_hidden_state']  # (n_sent, n_tokens, feat_dim)

            masks = []
            # token_masks = torch.tensor(sent['input_mask'])
            token_masks = torch.eye(input_ids.shape[1], device=self.device)
            # print("#####tokens", token_masks.shape)
            
            # m, n = token_masks.shape
            # if m!=n:
            #     print(m, n)
            #     print("#####token_masks", token_masks)

            for _, event in enumerate(sent['output_event']):
                this_mask = token_masks[event['tokens_number']]
                # print("####1", this_mask.shape)
                this_mask = torch.mean(this_mask, dim=0, keepdim=True)
                # print("#####2", this_mask.shape)
                masks.append(this_mask)
            for _,entity in enumerate(sent['output_entity']):
                this_mask = torch.mean(token_masks[entity['tokens_number']], dim=0, keepdim=True)
                masks.append(this_mask)
                
            masks = torch.cat(masks, dim=0).cuda()
            # print("#size", masks.shape, encoder_hidden_state.shape)
            # masks = torch.unsqueeze(masks, dim=0)
            encoder_hidden_state = encoder_hidden_state.squeeze()
            # print("#size2", masks.shape, encoder_hidden_state.shape)
            
            features.append(masks @ encoder_hidden_state)

        features = torch.cat(features)    #dim=0 1? encoder_hidden_state * input_mask = 所求表征
        return self.gsl(features, adj)  # gae, only encoder
    