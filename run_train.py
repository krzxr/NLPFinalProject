from model import *
from train import *
import os
#must modify model's num_labels as well
input_file_name = 'train_80_test_10_valid_top_2.pkl'
epochs = 5
lr = 5e-4
attempt = 2
name = input_file_name.split('.')[0]+'_epochs_'+str(epochs)+"_attempt_"+str(attempt)
if not os.path.exists('./logs/'+name):
    os.mkdir('./logs/'+name)
if not os.path.exists('./results/'+name):
    os.mkdir('./results/'+name)
train,test = get_train_test(input_file_name)
trainer_finetune(name,epochs,lr,optimizer,model,train,test)
