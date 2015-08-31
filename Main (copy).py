import theano.tensor as T
import theano,os
import numpy as np
import matplotlib.pyplot as plt
from Layers import hidden,lstm,gru,BiDirectionLSTM,decoder,BiDirectionGRU
from Models import ENC_DEC
from preprocess import Tokenizer
from initializations import prepare_data

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from konlpy.tag import Twitter
from konlpy.utils import pprint


def korean_morph(text):
    twitter = Twitter()
    
    s=twitter.morphs(str(unicode(text)))
    
    s=' '.join(s)
    
    
    
    return s

print 'Initializing model...'

#theano.config.exception_verbosity='high'

#theano.config.optimizer='None' 

def sampling(i,model,input,output,seq,seq_mask,targets,stochastic,n_gen_maxlen,n_words):
    test=seq[:,i]
    test_mask=seq_mask[:,i]
    
    truth=targets[:,i]
    
    guess = model.gen_sample(test,test_mask,stochastic)
    
    print 'Input: ',' '.join(input.sequences_to_text(test))
    
    print 'Truth: ',' '.join(output.sequences_to_text(truth))
    
    prob=np.asarray(guess[0],dtype=np.float)
    
    estimate=guess[1]
    
    print 'Sample: ',' '.join(output.sequences_to_text(estimate))
    
    return prob,estimate

n_epochs = 50
lr=0.001
momentum_switchover=5
learning_rate_decay=0.999
optimizer="RMSprop"

snapshot_Freq=50
sample_Freq=15
val_Freq=50

n_sentence=9000
n_batch=128 
n_chapter=None ## unit of slicing corpus
n_maxlen=100 ##max length of sentences in tokenizing
n_gen_maxlen=20 ## max length of generated sentences
n_words_x=10000 ## max numbers of words in dictionary
n_words_y=10000 ## max numbers of words in dictionary
dim_word=400  ## dimention of word embedding 

n_u = dim_word
n_h = 1000 ## number of hidden nodes in encoder

n_d = 1000 ## number of hidden nodes in decoder
n_y = dim_word

stochastic=False
verbose=1



## tokenize text, change to matrix

text=[]
with open("data/TED2013.raw.en") as f:
    for line in f:
        text.append(line)
        #text.append(korean_morph(line))
input=Tokenizer(n_words)
input.fit_on_texts(text)
seq=input.texts_to_sequences(text,n_sentence,n_maxlen)

n_words_x=input.nb_words

text=[]
with open("data/TED2013.raw.en") as f:
    for line in f:
        text.append(line)

output=Tokenizer(n_words)
output.fit_on_texts(text)
targets=output.texts_to_sequences(text,n_sentence,n_maxlen)

n_words_y=output.nb_words

targets[:-1]=targets[1:]

seq,seq_mask,targets,targets_mask=prepare_data(seq,targets,n_maxlen)

####build model

mode='tr'

model = ENC_DEC(n_u,n_h,n_d,n_y,n_epochs,n_chapter,n_batch,n_gen_maxlen,n_words_x,n_words_y,dim_word,
            momentum_switchover,lr,learning_rate_decay,snapshot_Freq,sample_Freq)
model.add(BiDirectionGRU(n_u,n_h))
model.add(decoder(n_h,n_d,n_y))
model.build()



filepath='data/ted.pkl'

if mode=='tr':
    if os.path.isfile(filepath): model.load(filepath)
    
    
    model.train(seq,seq_mask,targets,targets_mask,input,output,verbose,optimizer)
    model.save(filepath)
    
    ##draw error graph 
    plt.close('all')
    fig = plt.figure()
    ax3 = plt.subplot(111)   
    plt.plot(model.errors)
    plt.grid()
    ax3.set_title('Training error')    
    plt.savefig('error.png')
    
    
elif mode=='te':
    if os.path.isfile(filepath): model.load(filepath)
    else: 
        raise IOError('loading error...')

    i=20
    for j in range(i):
        k=np.random.randint(1,n_sentence)
        a=j+1
        print('\nsample %i >>>>'%a)
        prob,estimate=sampling(k,model,input,output,seq,seq_mask,targets,stochastic,n_gen_maxlen,n_words)
 

 



    
