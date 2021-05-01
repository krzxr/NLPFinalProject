from transformers import AdamW, BertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments

import pickle
import torch

print('loading model')
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
print('complete loading model')


print("loading data")
input_file = 'train_test_ratio_80_top_10.pkl'
epochs=3
batches = 100
data = pickle.load(open(input_file,'rb'))
train = data['train']
test = data['test']
print("complete loading data")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
prev_num_samples = 0
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
print("start training")
for epoch in range(epochs):
    for batch in range(batches+1):
        if batch==batches:
            num_samples = len(train)
        else:
            num_samples = len(train)//batches * (batch+1)
        if prev_num_samples == num_samples:
            break
        print("epoch",epoch,"batch",batch)
        print("encoding, num sample:",num_samples-prev_num_samples)
        batch_train = train[prev_num_samples:num_samples]
        batch_test = test[prev_num_samples:num_samples]
        train_texts = [instance[0] for instance in batch_train]
        train_labels = torch.tensor([int(instance[1]) for instance in batch_train]).unsqueeze(0)
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
        prev_num_samples = num_samples
model.eval()

