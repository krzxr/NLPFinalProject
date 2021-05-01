from torch.utils.data import DataLoader
from transformers import AdamW, BertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments

import pickle
import torch


    


class BlogsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([[self.labels[idx]]])
        #print(item)
        #x = 2/0
        return item

    def __len__(self):
        return len(self.labels)



def get_train_test(file):
    data = pickle.load(open(file,'rb'))
    train = data['train']
    test = data['test']
    # print(len(train))
    # print(train[1])

    train_texts = [instance[0] for instance in train]
    #train_labels = torch.tensor([int(instance[1]) for instance in train]).unsqueeze(0)
    train_labels = [int(instance[1]) for instance in train]    
    test_texts = [instance[0] for instance in test]
    #test_labels = torch.tensor([int(instance[1]) for instance in test]).unsqueeze(0)
    test_labels = [int(instance[1]) for instance in test]    

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    #train_encodings = tokenizer(train_texts, return_tensors='pt',truncation=True, padding=True)
    #test_encodings = tokenizer(test_texts, return_tensors = 'pt', truncation=True, padding=True)
    train_encodings = tokenizer(train_texts,truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = BlogsDataset(train_encodings, train_labels)
    test_dataset = BlogsDataset(test_encodings, test_labels)
    return train_dataset, test_dataset

train_dataset, test_dataset = get_train_test('train_test_ratio_80_top_10.pkl')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
for param in model.base_model.parameters():
    param.requires_grad = False
model.train()

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(model.parameters(), lr=1e-5)
def trainer_finetune():
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
    )
    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments
        train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset            # evaluation dataset
    )
    trainer.train()
    trainer.evaluate()

def torch_finetune():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    for epoch in range(3):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
    model.eval()
def finetune(input_file,epochs):
    data = pickle.load(open(input_file,'rb'))
    train = data['train']
    test = data['test']

    train_texts = [instance[0] for instance in train]
    train_labels = torch.tensor([int(instance[1]) for instance in train]).unsqueeze(0)
    test_texts = [instance[0] for instance in test]
    test_labels = torch.tensor([int(instance[1]) for instance in test]).unsqueeze(0)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_encodings = tokenizer(train_texts, return_tensors='pt',truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, return_tensors = 'pt', truncation=True, padding=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    for epoch in range(epochs):
        if True:
            print("epoch",epoch)
            optimizer.zero_grad()
            input_ids = train_encodings['input_ids'].to(device)
            attention_mask = train_encodings['attention_mask'].to(device)
            labels = train_labels['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(loss)
    model.eval()
finetune('train_test_ratio_80_top_10.pkl',3)
# torch_finetune()
