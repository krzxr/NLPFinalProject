import os
import random
from lxml import html
import random
import pickle
random.seed(0)
path = './../blogs/'
def get_authors_n_blogs():
  files = os.listdir(path)
  ls = []
  for fname in files:
    with open(path+fname,'r',encoding='latin1') as fp:
        f = fp.read()
    f = html.fromstring(f.encode('utf-8'))
    blogs = f.findall('post')
    blogs = list(filter(lambda s:len(text_normalize(s.text))>10,blogs))
    ls.append((fname, len(blogs)))
  ls.sort(key = lambda f: f[1], reverse = True)
  pickle.dump(ls, open('authors-n_blogs.pkl','wb'))
  
text_normalize = lambda text: ' '.join(list(filter(lambda x:x!='\n',text.split())))
def get_train_test(ratio, top_k):
  author_files = [ item[0] for item in pickle.load(open('authors-n_blogs.pkl','rb'))[:top_k] ]
  print('complete top_k retrival')
  train, test, valid = [], [], []
  for i,fname in enumerate(author_files):
    author = fname.split('.')[0]
    with open(path+fname,'r',encoding='latin1') as fp:
        f = fp.read()
    f = html.fromstring(f.encode('utf-8'))
    blogs = f.findall('post')
    print("blog len",len(blogs))
    blogs = list(filter(lambda s:len(text_normalize(s.text))>10,blogs))
    print("blog len",len(blogs))
    if len(blogs)<2:
      print("insufficent blog length!")

    n_test = max(int(len(blogs)* (1-ratio)*1/2),1)
    n_valid = max(int(len(blogs) * (1-ratio)),2)
    random.shuffle(blogs)

    test_from_f = blogs[:n_test]
    valid_from_f = blogs[n_test:n_valid]
    train_from_f = blogs[n_valid:]
    
    train_from_f = [(text_normalize(item.text) ,author) for item in train_from_f]
    test_from_f = [( text_normalize(item.text) ,author) for item in test_from_f]
    valid_from_f = [( text_normalize(item.text) ,author) for item in valid_from_f]
    train+=train_from_f
    test+=test_from_f
    valid+=valid_from_f
    print(f'complete file %d split, train/test: %d/%d'%(i,len(blogs)-n_test,n_test))
  pickle.dump({'train':train,'test':test,'valid':valid}, open(f'train_%d_test_%d_valid_top_%d.pkl'%(round(ratio*100), round((1-ratio)/2*100),top_k),'wb'))
  #return train, test



