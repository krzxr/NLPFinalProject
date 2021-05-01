from torch.utils.data import DataLoader
from transformers import AdamW, BertForSequenceClassification, DistilBertTokenizerFast, BertTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle
import torch
import random

class BlogsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        labels_set = list(set(self.labels))
        labels_set.sort()
        self.labels_map = {label:i for i,label in enumerate(labels_set)}
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels_map[self.labels[idx]]
        return item

    def __len__(self):
        return len(self.labels)


def get_train_test(input_file):
    data = pickle.load(open(input_file,'rb'))
    train = data['train']
    random.shuffle(train)
    test = data['valid']
    random.shuffle(test)
    # print(len(train))
    # print(train[1])

    train_texts = [instance[0] for instance in train]
    #train_labels = torch.tensor([int(instance[1]) for instance in train]).unsqueeze(0)
    train_labels = [int(instance[1]) for instance in train]    
    test_texts = [instance[0] for instance in test]
    #test_labels = torch.tensor([int(instance[1]) for instance in test]).unsqueeze(0)
    test_labels = [int(instance[1]) for instance in test]    

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    train_encodings = tokenizer(train_texts, return_tensors='pt',truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, return_tensors = 'pt', truncation=True, padding=True)
    #train_encodings = tokenizer(train_texts,truncation=True, padding=True)
    #test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = BlogsDataset(train_encodings, train_labels)
    test_dataset = BlogsDataset(test_encodings, test_labels)

    return train_dataset, test_dataset
def trainer_finetune(name,epochs,lr,optimizer,model,train_dataset,test_dataset):
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    training_args = TrainingArguments(
        output_dir='./results/'+name,          # output directory
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=8,   # batch size for evaluation
        learning_rate = lr,
        weight_decay = 0,
        adam_beta1=.9,
        adam_beta2=.999,
        adam_epsilon = 1e-8,
        max_grad_norm = 1,
        num_train_epochs=epochs,              # total # of training epochs
        warmup_steps = 100,
        logging_steps = 500,
        logging_dir='./logs/'+name,    # directory for storing logs
        save_steps = 1000,
    )
    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments
        train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset,         # evaluation dataset
        compute_metrics=compute_metrics,
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
def finetune(optimizer, model,input_file = 'train_test_ratio_80_top_10.pkl',epochs=3,batches = 1000):
    print("start loading")
    data = pickle.load(open(input_file,'rb'))
    train = data['train']
    labels = list(set([item[1] for item in train]))
    labels.sort()
    labels = {label:i for i,label in enumerate(labels)}
    test = data['test']
    
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    prev_end_idx = 0
    print("start training")
    for epoch in range(epochs):
        for batch in range(batches+1):
            if batch==batches:
                end_idx = len(train)
            else:
                end_idx = len(train)//batches * (batch+1)
            if prev_end_idx == end_idx:
                break
            print("epoch",epoch,"batch",batch)
            print("encoding, num sample:",end_idx-prev_end_idx)
            batch_train = train[prev_end_idx:end_idx]
            batch_test = test[prev_end_idx:end_idx]
            train_texts = [instance[0] for instance in batch_train]
            train_labels = torch.tensor([ labels[instance[1]] for instance in batch_train]).unsqueeze(0)
            #test_texts = [instance[0] for instance in batch_test]
            #test_labels = torch.tensor([int(instance[1]) for instance in batch_test]).unsqueeze(0)


            train_encodings = tokenizer(train_texts, return_tensors='pt',truncation=True, padding=True)
            #test_encodings = tokenizer(test_texts, return_tensors = 'pt', truncation=True, padding=True)

            optimizer.zero_grad()
            print("training")
            input_ids = train_encodings['input_ids'].to(device)
            attention_mask = train_encodings['attention_mask'].to(device)
            labels = train_labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(loss)
            prev_end_idx = end_idx
    model.eval()
# torch_finetune()
