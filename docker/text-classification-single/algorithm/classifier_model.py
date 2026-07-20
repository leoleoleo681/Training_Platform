# coding=utf-8

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from loss_function import AsymmetricLoss, FocalLoss


class BertForMultiLabel(BertPreTrainedModel):
    """
    支持单、多标签分类模型；
    支持ASL、Focal损失；
    """
    def __init__(self, config, label_distribution=None, distribution_gamma=0, alpha=2, pool_type='first-last-avg',focal_gamma=0):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.pool_type = pool_type
        self.label_distribution = label_distribution # tensor
        self.alpha = alpha
        self.distribution_gamma = distribution_gamma
        self.focal_gamma = focal_gamma
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None
    ):
        """
        Args:
            input_ids: torch.LongTensor of shape (batch_size, seq_length)
            attention_mask: torch.LongTensor of shape (batch_size, seq_length)
            labels: (Optional) torch.LongTensor of shape (batch_size, num_labels)
        """
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        if self.pool_type == 'cls':
            sequence_output = bert_outputs[0][:,0]
        elif self.pool_type == "last-avg":
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            last_hidden_state = bert_outputs[2][-1]
            mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
            token_count = mask.sum(dim=1).clamp(min=1.0)
            sequence_output = (last_hidden_state * mask).sum(dim=1) / token_count
        elif self.pool_type == 'first-last-avg':
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            mask = attention_mask.unsqueeze(-1).to(dtype=bert_outputs[2][1].dtype)
            token_count = mask.sum(dim=1).clamp(min=1.0)
            first_avg = (bert_outputs[2][1] * mask).sum(dim=1) / token_count
            last_avg = (bert_outputs[2][-1] * mask).sum(dim=1) / token_count
            sequence_output = (first_avg+last_avg)/2
        else:
            raise ValueError("不支持的pool_type: {}".format(self.pool_type))

        pooled_output = self.dense(sequence_output)
        pooled_output = self.activation(pooled_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits, pooled_output)
        
        if labels is not None:
            if self.loss_type == "BCE":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).to(torch.float))

            # 基于BCE 服务多标签
            elif self.loss_type == "ASL":
                loss_fct = AsymmetricLoss(
                    label_distribution = self.label_distribution,
                    distribution_gamma = self.distribution_gamma,
                    alpha = self.alpha,
                    gamma_neg = 1,
                    gamma_pos = 0
                )
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).to(torch.float))
            
            elif self.loss_type == "CE":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            elif self.loss_type == "Focal":
                loss_fct = FocalLoss(
                    label_distribution = self.label_distribution,
                    alpha = self.alpha,
                    gamma = self.focal_gamma, # gamma 越大越压低易分类样本、关注难样本，但过大会受噪声影响
                    distribution_gamma = self.distribution_gamma, # distribution_gamma 越大越压低高频类别、相对照顾低频类别，但过大会降低主流类别表现
                )
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # DataParallel无法直接聚合各GPU返回的0维标量；统一返回长度为1的loss。
            outputs = (loss.unsqueeze(0),) + outputs

        return outputs
        
