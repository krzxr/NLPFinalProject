from torch.utils.data import DataLoader
from transformers import AdamW, BertForSequenceClassification, DistilBertTokenizerFast, BertTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle
import torch
import numpy as np
import os 
def compute(name,input_sentence):
    model_name = {'word':'shuffle_unigrams_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-12760',
          'normal':'single_normal_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-12760',
          'lower': 'lowercase_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-12760',
          'double': 'double_matches_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-8660',
          'double_lower': 'double_matches_lowercase_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-8660',
          'double_unigram':'double_matches_unigram_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-8227',
          'same_blog':'same_num_blogs_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-8660',
          'same_shuffle':'same_num_shuffle_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-8660/',
          'same_lowercase':'same_num_lowercase_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-8660/',
          'same_no_pun':'same_num_no_pun_2authors_attempt_5_epochs_20_lr_0.005/checkpoint-8660'}

    path = './results/'+ model_name[name]
    name = name.split('/')[0]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = BertForSequenceClassification.from_pretrained(path,local_files_only=True, num_labels= 2).to(device)

    #model = BertForSequenceClassification.from_pretrained('bert-base-cased', \
    #    num_labels = 10)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model.eval()
    
    print("start evaluation")
    batch_size = 4
    prev_batch = 0
    errors = []
    alls = []
    alls_text=[]
    errors_text=[]
    y_preds, y_labels = [],[]
    idx=0
    texts = [input_sentence]
    encodings = tokenizer(texts, return_tensors='pt',truncation=True, padding=True)

    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs[0].to(torch.device('cpu')).detach().numpy()
    preds = [np.argmax(pred) for pred in logits]
    e_logits = np.exp(logits)
    probs = e_logits/np.sum(e_logits,1).reshape((-1,1))
    print("preds",preds,probs)
    del outputs
    #pickle.dump(errors, open('errors.pkl','wb'))
    
def cross_model_analysis(name1,name2,file_name):
    alls1 = pickle.load(open("./pkl/"+name1+'.pkl','rb'))
    alls2 = pickle.load(open("./pkl/"+name2+'.pkl','rb'))
    
    if not file_name[:-1] in os.listdir('./cross-txt/'):
        os.mkdir('./cross-txt/'+file_name)
    
    with open("./cross-txt/"+file_name+"all_w_text.txt","w") as fp:
        for tup1,txt1,tup2,txt2 in zip(alls1[0],alls1[1],alls2[0],alls2[1]):
        
            idx,pred, label, logit, prob = tup1
            idx2,pred2, label2, logit2, prob2 = tup2
            s = 'id '+str(idx)+' pred '+str(pred)+' label ' +str(label) + ' logit '+str(logit[0])+ ' '+str(logit[1])+' prob ' +str(prob[0])+' ' +str(prob[1])+'\n'
            s += ' pred2 '+str(pred2)+ ' logit2 '+str(logit2[0])+ ' '+str(logit2[1])+' prob2 ' +str(prob2[0])+' ' +str(prob2[1])+'\n'
            fp.write(s)
            fp.write(txt1+'\n\n')
    with open("./cross-txt/"+file_name+"error_w_text.txt","w") as fp:
        for tup1,txt1,tup2,txt2 in zip(alls1[0],alls1[1],alls2[0],alls2[1]):
            
            idx,pred, label, logit, prob = tup1
            idx2,pred2, label2, logit2, prob2 = tup2
            if pred==label and pred2==label2:
                continue
            s = 'id '+str(idx)+' pred '+str(pred)+' label ' +str(label) + ' logit '+str(logit[0])+ ' '+str(logit[1])+' prob ' +str(prob[0])+' ' +str(prob[1])+'\n'
            s += ' pred2 '+str(pred2)+ ' logit2 '+str(logit2[0])+ ' '+str(logit2[1])+' prob2 ' +str(prob2[0])+' ' +str(prob2[1])+'\n'
            fp.write(s)
            fp.write(txt1+'\n\n')
        
    with open("./cross-txt/"+file_name+"diff_w_text.txt","w") as fp:
        for tup1,txt1,tup2,txt2 in zip(alls1[0],alls1[1],alls2[0],alls2[1]):
            
            idx,pred, label, logit, prob = tup1
            idx2,pred2, label2, logit2, prob2 = tup2
            if pred==pred2:
                continue
            s = 'id '+str(idx)+' pred '+str(pred)+' label ' +str(label) + ' logit '+str(logit[0])+ ' '+str(logit[1])+' prob ' +str(prob[0])+' ' +str(prob[1])+'\n'
            s += ' pred2 '+str(pred2)+ ' logit2 '+str(logit2[0])+ ' '+str(logit2[1])+' prob2 ' +str(prob2[0])+' ' +str(prob2[1])+'\n'
            fp.write(s)
            fp.write(txt1+'\n\n')



      

def error_detection(file_name, input_file = './datasets/shuffle_unigrams_2authors/shuffle_unigrams_2authors_t80t10v10.pkl', name = 'shuffle_unigrams_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-12760'):
    print("start preparation "+file_name)
    if not file_name[:-1] in os.listdir('./txt/'):
        os.mkdir('./txt/'+file_name)
    # num_labels = int(input_file.split('.')[0].split('_')[-1])
    num_labels = 2
    datatype = input_file.split('/')[1]
    data = pickle.load(open(input_file,'rb'))
    train = data['train']
    labels = list(set([item[1] for item in train]))
    labels.sort()
    labels_map = {label:i for i,label in enumerate(labels)}
    valid = data['valid']
    path = './results/'+name
    name = name.split('/')[0]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = BertForSequenceClassification.from_pretrained(path,local_files_only=True, num_labels= num_labels).to(device)

    #model = BertForSequenceClassification.from_pretrained('bert-base-cased', \
    #    num_labels = 10)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model.eval()
    
    print("start evaluation")
    dataset = valid
    batch_size = 4
    prev_batch = 0
    errors = []
    alls = []
    alls_text=[]
    errors_text=[]
    y_preds, y_labels = [],[]
    idx=0
    for batch in range(batch_size,len(dataset),batch_size):
        batch_data = dataset[prev_batch:batch]
        texts = [instance[0] for instance in batch_data]
        encodings = tokenizer(texts, return_tensors='pt',truncation=True, padding=True)

        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0].to(torch.device('cpu')).detach().numpy()
        labels = [ labels_map[instance[1]] for instance in batch_data]
        y_labels+=labels
        #print(labels)
        preds = [np.argmax(pred) for pred in logits]
        y_preds+=list(preds)
        e_logits = np.exp(logits)
        probs = e_logits/np.sum(e_logits,1).reshape((-1,1))

        for pred, logit, prob,label,txt in zip(preds,logits,probs, labels,texts):
            if pred != label:
                #print('pred',pred,'label',label,'logit',logit,'prob',prob)
                error = (idx,pred,label,logit,prob)
                errors.append(error)
                errors_text.append(txt)
            alls.append((idx,pred,label,logit,prob))
            alls_text.append(txt)
            idx+=1
        #print(outputs,labels,prob)

        prev_batch = batch
        del outputs
    #pickle.dump(errors, open('errors.pkl','wb'))
    
    with open("./txt/"+file_name+"error_w_text.txt","w") as fp:
        print('error w txt')
        for error,txt in zip(errors,errors_text):
            idx,pred, label, logit, prob = error
            s = 'id '+str(idx)+' pred '+str(pred)+' label ' +str(label) + ' logit '+str(logit[0])+ ' '+str(logit[1])+' prob ' +str(prob[0])+' ' +str(prob[1])+'\n'
            fp.write(s)
            fp.write(txt+'\n\n')
    
    with open("./txt/"+file_name+"all_w_text.txt","w") as fp:
        print('all w txt')
        for tup,txt in zip(alls,alls_text):
            idx,pred, label, logit, prob = tup
            s = 'id '+str(idx)+' pred '+str(pred)+' label ' +str(label) + ' logit '+str(logit[0])+ ' '+str(logit[1])+' prob ' +str(prob[0])+' ' +str(prob[1])+'\n'
            fp.write(s)
            fp.write(txt+'\n\n')
    with open("./txt/"+file_name+"error.txt","w") as fp:
        print('error')
        for error,txt in zip(errors,errors_text):
            idx,pred, label, logit, prob = error
            s = 'id '+str(idx)+' pred '+str(pred)+' label ' +str(label) + ' logit '+str(logit[0])+ ' '+str(logit[1])+' prob ' +str(prob[0])+' ' +str(prob[1])+'\n'
            fp.write(s)
    with open("./txt/"+file_name+"all.txt","w") as fp:
        print('all')
        for tup,txt in zip(alls,alls_text):
            idx,pred, label, logit, prob = tup
            s = 'id '+str(idx)+' pred '+str(pred)+' label ' +str(label) + ' logit '+str(logit[0])+ ' '+str(logit[1])+' prob ' +str(prob[0])+' ' +str(prob[1])+'\n'
            fp.write(s)
    with open("./txt/"+file_name+"close_error_w_text.txt","w") as fp:
        print('close error w txt')
        for error,txt in zip(errors,errors_text):
            idx,pred, label, logit, prob = error
            s = 'id '+str(idx)+' pred '+str(pred)+' label ' +str(label) + ' logit '+str(logit[0])+ ' '+str(logit[1])+' prob ' +str(prob[0])+' ' +str(prob[1])+'\n'
            if abs(prob[0]-prob[1])<.05:
                fp.write(s)
                fp.write(txt+'\n\n')
    with open("./txt/"+file_name+"far_error_w_text.txt","w") as fp:
        print('far w txt')
        for error,txt in zip(errors,errors_text):
            idx,pred, label, logit, prob = error
            s = 'id '+str(idx)+' pred '+str(pred)+' label ' +str(label) + ' logit '+str(logit[0])+ ' '+str(logit[1])+' prob ' +str(prob[0])+' ' +str(prob[1])+'\n'
            if abs(prob[0]-prob[1])>.5:
                fp.write(s)
                fp.write(txt+'\n\n')

    s = precision_recall_fscore_support(y_labels,y_preds)
    with open("./txt/"+file_name+"metrics.txt","w") as fp:
        print("metrics")
        for i,hint in enumerate(['precision','recall','fbeta','num samples']):
            fp.write(hint+' ' +str(s[i])+' avg: '+str((s[i][0]+s[i][1])/2)+'\n')
    pickle.dump([alls,alls_text],open('./pkl/'+file_name[:-1]+'.pkl','wb'))
    #return errors, alls

