from torch.utils.data import DataLoader
from transformers import AdamW, BertForSequenceClassification, DistilBertTokenizerFast, BertTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle
import torch
import numpy as np
def error_detection(input_file = 'train_80_test_10_valid_top_2.pkl'):
    print("start preparation")
    num_labels = int(input_file.split('.')[0].split('_')[-1])
    data = pickle.load(open(input_file,'rb'))
    train = data['train']
    labels = list(set([item[1] for item in train]))
    labels.sort()
    labels_map = {label:i for i,label in enumerate(labels)}
    valid = data['valid']
    path = './results/train_80_test_10_valid_top_2_epochs_5_attempt_1/checkpoint-3000'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = BertForSequenceClassification.from_pretrained(path,local_files_only=True, num_labels= num_labels).to(device)

    #model = BertForSequenceClassification.from_pretrained('bert-base-cased', \
    #    num_labels = 10)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model.eval()
    
    print("start evaluation")
    dataset = valid
    batch_size = 8
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
        print(labels)
        preds = [np.argmax(pred) for pred in logits]
        y_preds+=list(preds)
        e_logits = np.exp(logits)
        probs = e_logits/np.sum(e_logits,1).reshape((-1,1))

        for pred, logit, prob,label,txt in zip(preds,logits,probs, labels,texts):
            if pred != label:
                print('pred',pred,'label',label,'logit',logit,'prob',prob)
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
   
    print('error w txt')
    with open("./txt/error_w_text.txt","w") as fp:
        for error,txt in zip(errors,errors_text):
            idx,pred, label, logit, prob = error
            s = 'id '+str(idx)+' pred '+str(pred)+' label ' +str(label) + ' logit '+str(logit[0])+ ' '+str(logit[1])+' prob ' +str(prob[0])+' ' +str(prob[1])+'\n'
            fp.write(s)
            fp.write(txt+'\n\n')
    
    print('all w txt')
    with open("./txt/all_w_text.txt","w") as fp:
        for tup,txt in zip(alls,alls_text):
            idx,pred, label, logit, prob = tup
            s = 'id '+str(idx)+' pred '+str(pred)+' label ' +str(label) + ' logit '+str(logit[0])+ ' '+str(logit[1])+' prob ' +str(prob[0])+' ' +str(prob[1])+'\n'
            fp.write(s)
            fp.write(txt+'\n\n')
    print('error')
    with open("./txt/error.txt","w") as fp:
        for error,txt in zip(errors,errors_text):
            idx,pred, label, logit, prob = error
            s = 'id '+str(idx)+' pred '+str(pred)+' label ' +str(label) + ' logit '+str(logit[0])+ ' '+str(logit[1])+' prob ' +str(prob[0])+' ' +str(prob[1])+'\n'
            fp.write(s)
    print('all')
    with open("./txt/all.txt","w") as fp:
        for tup,txt in zip(alls,alls_text):
            idx,pred, label, logit, prob = tup
            s = 'id '+str(idx)+' pred '+str(pred)+' label ' +str(label) + ' logit '+str(logit[0])+ ' '+str(logit[1])+' prob ' +str(prob[0])+' ' +str(prob[1])+'\n'
            fp.write(s)
    print('close error w txt')
    with open("./txt/close_error_w_text.txt","w") as fp:
        for error,txt in zip(errors,errors_text):
            idx,pred, label, logit, prob = error
            s = 'id '+str(idx)+' pred '+str(pred)+' label ' +str(label) + ' logit '+str(logit[0])+ ' '+str(logit[1])+' prob ' +str(prob[0])+' ' +str(prob[1])+'\n'
            if abs(prob[0]-prob[1])<.05:
                fp.write(s)
                fp.write(txt+'\n\n')
    print('far w txt')
    with open("./txt/far_error_w_text.txt","w") as fp:
        for error,txt in zip(errors,errors_text):
            idx,pred, label, logit, prob = error
            s = 'id '+str(idx)+' pred '+str(pred)+' label ' +str(label) + ' logit '+str(logit[0])+ ' '+str(logit[1])+' prob ' +str(prob[0])+' ' +str(prob[1])+'\n'
            if abs(prob[0]-prob[1])>.5:
                fp.write(s)
                fp.write(txt+'\n\n')
    print(precision_recall_fscore_support(y_labels,y_preds))
    return errors,alls

