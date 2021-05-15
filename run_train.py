from model import *
from train import *
import os
import argparse
#must modify model's num_labels as well
parser = argparse.ArgumentParser(description='train')
parser.add_argument("--file", "-f", required=True,
                        help="filename")
parser.add_argument("--lr", "-lr", required=False, default = 5e-3, type=float,
                        help="lr, default 5e-3")
parser.add_argument("--attempt", "-a", default=0,required=False,help='attempt number')
parser.add_argument("--epochs", "-e", default=5,required=False,type=int,help='num epoch')
args = parser.parse_args()
fname = args.file+'_2authors'
fdir = './datasets/'+fname+'/'+fname+'_t80t10v10.pkl'
print('\n'+fdir+'\n')
epochs = int(args.epochs)
lr = float(args.lr)
attempt = int(args.attempt)
'''
sentences_shuffle = False
words_shuffle = True
'''
name = fname+"_attempt_"+str(attempt)+'_epochs_'+str(epochs)+'_lr_'+str(lr)
print('\nname',name+'\n')
#name += "_shuffle_sent" if sentences_shuffle else ""
#name += "_shuffle_word" if words_shuffle else ""
if not os.path.exists('./logs/'+name):
    os.mkdir('./logs/'+name)
if not os.path.exists('./results/'+name):
    os.mkdir('./results/'+name)
train,test = get_train_test(fdir)
trainer_finetune(name,epochs,lr,optimizer,model,train,test)
