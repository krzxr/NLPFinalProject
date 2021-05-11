from transformers import BertTokenizer
import pickle
import torch
import random

random.seed(0)

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

def shuffle_sentences(text):
    L = text.split('.')
    random.shuffle(L)
    return '. '.join(L) + '.'

def shuffle_words(text):
    L = text.split(' ')
    random.shuffle(L)
    return ' '.join(L)

def get_train_test(input_file, sentences_shuffle=False, words_shuffle=False):
    data = pickle.load(open(input_file,'rb'))
    train = data['train']
    random.shuffle(train)
    test = data['valid']
    random.shuffle(test)
    # print(len(train))
    # print(train[1])

    # print(train[6][0])

    if sentences_shuffle:
        train = [(shuffle_sentences(instance[0]), instance[1]) for instance in train]
        test = [(shuffle_sentences(instance[0]), instance[1]) for instance in test]

    if words_shuffle:
        train = [(shuffle_words(instance[0]), instance[1]) for instance in train]
        test = [(shuffle_words(instance[0]), instance[1]) for instance in test]

    train_texts = [instance[0] for instance in train]
    #train_labels = torch.tensor([int(instance[1]) for instance in train]).unsqueeze(0)
    train_labels = [int(instance[1]) for instance in train]    
    test_texts = [instance[0] for instance in test]
    #test_labels = torch.tensor([int(instance[1]) for instance in test]).unsqueeze(0)
    test_labels = [int(instance[1]) for instance in test]

    # print(train_texts[6])   

    # for i in range(10):
    #     print(len(train[i][0].split('.')))

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    train_encodings = tokenizer(train_texts, return_tensors='pt',truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, return_tensors = 'pt', truncation=True, padding=True)
    #train_encodings = tokenizer(train_texts,truncation=True, padding=True)
    #test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = BlogsDataset(train_encodings, train_labels)
    test_dataset = BlogsDataset(test_encodings, test_labels)

    #return train_dataset, test_dataset