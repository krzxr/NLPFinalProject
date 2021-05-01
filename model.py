from transformers import AdamW, BertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments

import pickle
import torch


#path = './results/train_80_test_10_valid_top_10_epochs_5_attempt_5/checkpoint-5500'
#model = BertForSequenceClassification.from_pretrained(path,local_files_only=True, num_labels=10)
model = BertForSequenceClassification.from_pretrained('bert-base-cased', \
    num_labels = 2)
for param in model.base_model.parameters():
    param.requires_grad = False
model.train()
'''
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
'''
optimizer = AdamW(model.parameters())
