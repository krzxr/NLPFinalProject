from model import *
from train import *
import os
#must modify model's num_labels as well
input_file_name = 'train_80_test_10_valid_top_2.pkl'
epochs = 5
lr = 5e-4
attempt = 3
sentences_shuffle = False
words_shuffle = True
name = input_file_name.split('.')[0]+'_epochs_'+str(epochs)+"_attempt_"+str(attempt)
name += "_shuffle_sent" if sentences_shuffle else ""
name += "_shuffle_word" if words_shuffle else ""
if not os.path.exists('./logs/'+name):
    os.mkdir('./logs/'+name)
if not os.path.exists('./results/'+name):
    os.mkdir('./results/'+name)
train,test = get_train_test(input_file_name, sentences_shuffle, words_shuffle)
trainer_finetune(name,epochs,lr,optimizer,model,train,test)
