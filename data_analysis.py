import pickle
import numpy as np
from matplotlib import pyplot as plt
def get_distribution(input_file='train_80_test_10_valid_top_2.pkl'):
    data = pickle.load(open(input_file,'rb'))
    train = data['train']
    valid = data['valid']
    
    valid_author = {}
    valid_author_all = {}
    for text,author in valid: 
        text_len = len(text)
        if text_len<10:
            print(text)
        if text_len>20:
            valid_author[author] = valid_author.get(author,[]) + [text_len]
        valid_author_all[author] = valid_author_all.get(author,[]) + [text_len]

    for author in valid_author:
        array = np.array(valid_author[author])
        print(len(array),'/',len(valid_author_all[author]),int(array.mean()),int(np.median(array)),max(array),min(array))
        #plt.boxplot(array)
    #plt.savefig(input_file.split('.')[0]+'_valid_data_distribution.pdf')

