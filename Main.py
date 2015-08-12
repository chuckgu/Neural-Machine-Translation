import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from Layers import hidden,lstm,gru,BiDirectionLSTM,decoder
from Models import ENC_DEC
from preprocess import Tokenizer

print 'Initializing model...'

#theano.config.exception_verbosity='high'


n_u = 500
n_h = 1000

n_d = 1000
n_y = 500

n_epochs = 10   
lr=0.00095
n_sentence=9000

n_batch=128
n_chapter=3000
n_maxlen=15
n_words=500

text=[]
with open("data\OpenSubtitles2012.en-ko.en") as f:
    for line in f:
        text.append(line)
input=Tokenizer(n_words)
input.fit_on_texts(text)
seq=input.texts_to_matrix(text,"onehot",n_sentence)

time_steps_x=input.maxlen

text=[]
with open("data\OpenSubtitles2012.en-ko.en") as f:
    for line in f:
        text.append(line)

output=Tokenizer(n_words)
output.fit_on_texts(text)
targets=output.texts_to_matrix(text,"onehot",n_sentence)

time_steps_y=output.maxlen

targets[:-1]=targets[1:]

seq=seq.transpose(1,0,2)
targets=targets.transpose(1,0,2)




mode='tr'

model = ENC_DEC(n_u,n_h*2,n_d,n_y,lr,n_epochs,n_chapter,n_batch,n_maxlen)
model.add(BiDirectionLSTM(n_u,n_h))
model.add(decoder(n_h*2,n_d,n_y))


model.build()

i=16

if mode=='tr':
    model.load('data\encdec_enfr.pkl')
    
    
    model.train(seq,targets)
    model.save('data\encdec_enfr.pkl')
    
    plt.close('all')
    fig = plt.figure()
    ax1 = plt.subplot(311)
    plt.plot(seq[:,i])
    plt.grid()
    ax1.set_title('input')
    ax2 = plt.subplot(312)
    
    plt.scatter(xrange(time_steps_y), output.matrix_to_sequences(targets[:,i]), marker = 'o', c = 'b')
    plt.grid()
    
    test=seq[:,i][:,np.newaxis]
    
    guess = model.gen_sample(test)
    
    print output.matrix_to_text(targets[:,i])
    
    estimate=guess[1]
    print output.sequences_to_text(estimate)
    
    guess=np.asarray(guess[0],dtype=np.float).reshape((n_maxlen,n_y))
    
    
    
    
    guessed_probs = plt.imshow(guess.T, interpolation = 'nearest', cmap = 'gray')
    ax2.set_title('blue points: true class, grayscale: model output (white mean class)')
    
    ax3 = plt.subplot(313)
    plt.plot(model.errors)
    plt.grid()
    ax3.set_title('Training error')
    
elif mode=='te':
    model.load('data\encdec_enfr.pkl')
    
    
    plt.close('all')
    fig = plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(seq[:,i])
    plt.grid()
    ax1.set_title('input')
    ax2 = plt.subplot(212)
    t=targets[:,i]
    
    plt.scatter(xrange(time_steps_y), output.matrix_to_sequences(t), marker = 'o', c = 'b')
    plt.grid()
    
    test=seq[:,i][:,np.newaxis]
    
    guess = model.gen_sample(test)
    
    print output.matrix_to_text(t)
    
    estimate=guess[1]
    print output.sequences_to_text(estimate)
    
    guess=np.asarray(guess[0],dtype=np.float).reshape((n_maxlen,n_y))

    
    guessed_probs = plt.imshow(guess.T, interpolation = 'nearest', cmap = 'gray')
    ax2.set_title('blue points: true class, grayscale: model output (white mean class)')


