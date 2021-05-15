import os
from itertools import combinations 
import random
from lxml import html
import random
import pickle
random.seed(0)
path = './../blogs/'
def get_blogs():
  files = os.listdir(path)
  ls = []
  for fname in files:
    with open(path+fname,'r',encoding='latin1') as fp:
        f = fp.read()
    f = html.fromstring(f.encode('utf-8'))
    blogs = f.findall('post')
    blogs = list(filter(lambda s:len(text_normalize(s.text))>20,blogs))
    ls.append((fname, len(blogs)))
  ls.sort(key = lambda f: f[1], reverse = True)
  pickle.dump(ls, open('./datasets/blogs-n_blogs.pkl','wb'))
  
text_normalize = lambda text: ' '.join(list(filter(lambda x:x!='\n',text.split())))
def normalize_blogs(top_k):
  author_files = [ item[0] for item in pickle.load(open('./datasets/authors-n_blogs.pkl','rb'))[:top_k] ]
  all_blogs = []
  for i,fname in enumerate(author_files):
    author = fname.split('.')[0]
    with open(path+fname,'r',encoding='latin1') as fp:
        f = fp.read()
    f = html.fromstring(f.encode('utf-8'))
    blogs = f.findall('post')
    print("blog len",len(blogs))
    blogs = list(filter(lambda s:len(text_normalize(s.text))>20,blogs))
    print("blog len",len(blogs))
    if len(blogs)<2:
      print("insufficent blog length!")
    blogs = [(text_normalize(item.text) ,author) for item in blogs]
    random.shuffle(blogs)
    all_blogs.append(blogs)
  pickle.dump(all_blogs, open('./datasets/blogs_normalized_'+str(top_k)+'authors.pkl','wb'))
  return all_blogs

def split_and_dump(name,all_blogs):
  train,test,valid = [],[],[]
  ratio=.8
  for blogs in all_blogs:
    n_test = max(int(len(blogs)* (1-ratio)/2),1)
    n_valid = max(int(len(blogs) * (1-ratio)),2)
    test_from_f = blogs[:n_test]
    valid_from_f = blogs[n_test:n_valid]
    train_from_f = blogs[n_valid:]
    train+=train_from_f
    test+=test_from_f
    valid+=valid_from_f
  pickle.dump({'train':train,'test':test,'valid':valid}, open(f'%s_%dauthors_t%dt%dv%d.pkl'%(name,  len(all_blogs), 80, 10, 10),'wb'))
  return train, valid, test

# top 2 authors without any modification: this is the baseline
def single_normal_2authors_dataset():
  blogs = pickle.load(open('./datasets/blogs_normalized_2authors.pkl','rb'))
  name = 'single_normal'
  return split_and_dump('./datasets/'+name+'_2authors/'+name,blogs)

# every 2 authors among top 4 authors: this is used to verify the consistency 
# of performance under normal circumstances. 
def multiple_normal_2authors_dataset():
  blogs = pickle.load(open('./datasets/blogs_normalized_4authors.pkl','rb'))
  train, valid, test = [],[],[]
  for i, (blog1,blog2) in enumerate(combinations(blogs, 2)):
    train, valid, test = split_and_dump('./datasets/multiple_normal_2authors/'+str(i)+'th_normal',[blog1,blog2])
  return train, valid, test
# shuffle unigram on top 2 authors -- expect to decrease performance, which is reflective of the content performance decrease. shuffling unigram means removing bigrams and trigrams
def shuffle_unigrams_2authors_dataset():
  blogs = pickle.load(open('./datasets/blogs_normalized_2authors.pkl','rb'))
  name = 'shuffle_unigrams'
  new_blogs = []
  for blog in blogs:
    blog = [(shuffle_words(item[0]) ,item[1]) for item in blog]
    new_blogs.append(blog)
  return split_and_dump('./datasets/'+name+'_2authors/'+name,new_blogs)
  
# lowercase
def lowercase_2authors_dataset():
  blogs = pickle.load(open('./datasets/blogs_normalized_2authors.pkl','rb'))
  name = 'lowercase'
  new_blogs = []
  for blog in blogs:
    blog = [(lowercase(item[0]) ,item[1]) for item in blog]
    new_blogs.append(blog)
  return split_and_dump('./datasets/'+name+'_2authors/'+name,new_blogs)
# top 2 authors' number of blogs will be the same number 
def same_num_blogs_2authors_dataset():
  blogs = pickle.load(open('./datasets/blogs_normalized_2authors.pkl','rb'))
  blog1, blog2 = blogs 
  num_blogs = min(len(blog1),len(blog2))
  blogs = [blog1[:num_blogs],blog2[:num_blogs]]
  name = 'same_num_blogs'
  return split_and_dump('./datasets/'+name+'_2authors/'+name,blogs)

def same_num_lowercase_2authors_dataset():
  blogs = pickle.load(open('./datasets/blogs_normalized_2authors.pkl','rb'))
  blog1, blog2 = blogs 
  num_blogs = min(len(blog1),len(blog2))
  blogs = [blog1[:num_blogs],blog2[:num_blogs]]
  name = 'same_num_lowercase'
  new_blogs = []
  for blog in blogs:
    blog = [(lowercase(item[0]) ,item[1]) for item in blog]
    new_blogs.append(blog)
  return split_and_dump('./datasets/'+name+'_2authors/'+name,new_blogs)

def same_num_shuffle_2authors_dataset():
  blogs = pickle.load(open('./datasets/blogs_normalized_2authors.pkl','rb'))
  blog1, blog2 = blogs 
  num_blogs = min(len(blog1),len(blog2))
  blogs = [blog1[:num_blogs],blog2[:num_blogs]]
  name = 'same_num_shuffle'
  new_blogs = []
  for blog in blogs:
    blog = [(shuffle_words(item[0]) ,item[1]) for item in blog]
    new_blogs.append(blog)
  return split_and_dump('./datasets/'+name+'_2authors/'+name,new_blogs)
def same_num_no_pun_2authors_dataset():
  blogs = pickle.load(open('./datasets/blogs_normalized_2authors.pkl','rb'))
  blog1, blog2 = blogs 
  num_blogs = min(len(blog1),len(blog2))
  blogs = [blog1[:num_blogs],blog2[:num_blogs]]
  name = 'same_num_no_pun'
  new_blogs = []
  for blog in blogs:
    blog = [(no_pun(item[0]) ,item[1]) for item in blog]
    new_blogs.append(blog)
  return split_and_dump('./datasets/'+name+'_2authors/'+name,new_blogs)

# top 2 authors' # of blogs will be the same 
# match the len per blog for both bloggers 
def generate_double_matches_2authors():
  blogs = pickle.load(open('./datasets/blogs_normalized_2authors.pkl','rb'))
  blog1, blog2 = blogs 
  blog1.sort(key=lambda s:len(s[0]))
  blog2.sort(key=lambda s:len(s[0]))
  new_blog1, new_blog2 = [],[]
  num_blogs = min(len(blog1),len(blog2))
  blogs = [blog1[:num_blogs],blog2[:num_blogs]]
  for b1,b2 in zip(blog1,blog2):
    l = min(len(b1),len(b2))
    b1, b2 = b1[:l], b2[:l]
    new_blog1.append(b1)
    new_blog2.append(b2)
  return new_blog1, new_blog2

def double_matches_2authors():
  blogs = generate_double_matches_2authors()
  name = 'double_matches'
  return split_and_dump('./datasets/'+name+'_2authors/'+name,blogs)

def double_matches_unigram_2authors():
  blogs = generate_double_matches_2authors()
  new_blogs = []
  for blog in blogs:
    blog = [(lowercase(item[0]) ,item[1]) for item in blog]
    new_blogs.append(blog)
  name = 'double_matches_unigram'
  return split_and_dump('./datasets/'+name+'_2authors/'+name,new_blogs)
  
def double_matches_lowercase_2authors():
  blogs = generate_double_matches_2authors()
  new_blogs = []
  for blog in blogs:
    blog = [( shuffle_words(item[0]) ,item[1]) for item in blog]
    new_blogs.append(blog)
  name = 'double_matches_lowercase'
  return split_and_dump('./datasets/'+name+'_2authors/'+name,new_blogs)

# represent content 
def shuffle_words(text):
  L = text.split(' ')
  random.shuffle(L)
  return ' '.join(L)

def no_pun_2authors_dataset():
  blogs = pickle.load(open('./datasets/blogs_normalized_2authors.pkl','rb'))
  name = 'no_pun'
  new_blogs = []
  for blog in blogs:
    blog = [(no_pun(item[0]) ,item[1]) for item in blog]
    new_blogs.append(blog)
  return split_and_dump('./datasets/'+name+'_2authors/'+name,new_blogs)
def double_matches_no_pun_2authors():
  blogs = generate_double_matches_2authors()
  new_blogs = []
  for blog in blogs:
    blog = [( no_pun(item[0]) ,item[1]) for item in blog]
    new_blogs.append(blog)
  name = 'double_matches_no_pun'
  return split_and_dump('./datasets/'+name+'_2authors/'+name,new_blogs)
def no_pun(text):
  s = ''
  for i in text:
    if not i.isalnum() and not i==' ':
      s+=''
    else:
      s+=i
  return s

# represent style -- one of the most prominent/frequent characters of style 
def lowercase(text):
  return text.lower()
    
def shuffle_sentences(text):
  L = text.split('.')
  random.shuffle(L)
  return '. '.join(L) + '.'


