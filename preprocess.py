#-*- coding: utf-8 -*-
'''
    These preprocessing utils would greatly benefit
    from a fast Cython rewrite.
'''
from __future__ import absolute_import


import string, sys
import numpy as np
from six.moves import range
from six.moves import zip

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

def base_filter():
    f = string.punctuation
    f = f.replace("'", '')
    f += '\t\n'
    return f

def text_to_word_sequence(text, filters=base_filter(), lower=True, split=" "):
    '''prune: sequence of characters to filter out
    '''
    if lower:
        text = text.lower()
    text = text.translate(maketrans(filters, split*len(filters)))
    seq = text.split(split)
    return [_f for _f in seq if _f]


def one_hot(text, n, filters=base_filter(), lower=True, split=" "):
    seq = text_to_word_sequence(text, filters=filters, lower=lower, split=split)
    return [(abs(hash(w))%(n-1)+1) for w in seq]


class Tokenizer(object):
    def __init__(self, nb_words=None, filters=base_filter(), lower=True, split=" "):
        self.word_counts = {}
        self.word_docs = {}
        self.filters = filters
        self.split = split
        self.lower = lower
        self.nb_words = nb_words
        self.document_count = 0
        self.maxlen=0

    def fit_on_texts(self, texts):
        '''
            required before using texts_to_sequences or texts_to_matrix
            @param texts: can be a list or a generator (for memory-efficiency)
        '''
        self.document_count = 0
        
        for text in texts:
            self.document_count += 1
            seq = text_to_word_sequence(text, self.filters, self.lower, self.split)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                if w in self.word_docs:
                    self.word_docs[w] += 1
                else:
                    self.word_docs[w] = 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key = lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]
        self.word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc)+1)))))

        self.index_docs = {}
        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c


    def fit_on_sequences(self, sequences):
        '''
            required before using sequences_to_matrix 
            (if fit_on_texts was never called)
        '''
        self.document_count = len(sequences)
        self.index_docs = {}
        for seq in sequences:
            seq = set(seq)
            for i in seq:
                if i not in self.index_docs:
                    self.index_docs[i] = 1
                else:
                    self.index_docs[i] += 1


    def texts_to_sequences(self, texts,batch_size=None):
        '''
            Transform each text in texts in a sequence of integers.
            Only top "nb_words" most frequent words will be taken into account.
            Only words known by the tokenizer will be taken into account.

            Returns a list of sequences.
        '''
        if batch_size is None: batch_size=self.document_count
        res = []
        for vect in self.texts_to_sequences_generator(texts):
            res.append(vect)
            if len(res) >= batch_size:
                    break 
        return res

    def texts_to_sequences_generator(self, texts):
        '''
            Transform each text in texts in a sequence of integers.
            Only top "nb_words" most frequent words will be taken into account.
            Only words known by the tokenizer will be taken into account.

            Yields individual sequences.
        '''
        nb_words = self.nb_words
        for text in texts:
            seq = text_to_word_sequence(text, self.filters, self.lower, self.split)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if nb_words and i >= nb_words:
                        pass
                    else:
                        vect.append(i)
            yield vect


    def texts_to_matrix(self, texts, mode="onehot",batch_size=None,maxlen=None):
        '''
            modes: binary, count, tfidf, freq
        '''
        sequences = self.texts_to_sequences(texts,batch_size)
        return self.sequences_to_matrix(sequences, mode=mode,maxlen=maxlen)

    def sequences_to_matrix(self, sequences, mode="onehot",maxlen=None):
        '''
            modes: binary, count, tfidf, freq
        '''
        
        lengths = [len(s) for s in sequences]
        if maxlen is None:
            self.maxlen = np.max(lengths)
            print self.maxlen

        if not self.nb_words:
            if self.word_index:
                nb_words = len(self.word_index)
            else:
                raise Exception("Specify a dimension (nb_words argument), or fit on some text data first")
        else:
            nb_words = self.nb_words
        if mode == "tfidf" and not self.document_count:
            raise Exception("Fit the Tokenizer on some data before using tfidf mode")

        X = np.zeros((len(sequences),self.maxlen,nb_words))
        for i, seq in enumerate(sequences):
            if not seq:
                pass
            counts = {}
            position=np.zeros((nb_words,self.maxlen))
            k=0
            for j in seq:
                if k>=self.maxlen:break 
                if j >= nb_words:
                    pass
                if j not in counts:
                    counts[j] = 1.        
                else:
                    counts[j] += 1   
                position[j][k]=1.        
                k+=1
            for j, c in list(counts.items()):
                if mode == "count":
                    X[i][j] = c
                elif mode == "freq":
                    X[i][j] = c/len(seq)
                elif mode == "binary":
                    X[i][j] = 1    
                elif mode == "onehot":
                    for l,po in enumerate(position[j]):
                        if po:     
                            X[i][l][j] = 1
                elif mode == "tfidf":
                    tf = np.log(c/len(seq))
                    df = (1 + np.log(1 + self.index_docs.get(j, 0)/(1 + self.document_count)))
                    X[i][j] = tf / df
                else:
                    raise Exception("Unknown vectorization mode: " + str(mode))
        return X


    def sequences_to_text(self, sequences):

        vect = []
        for n, w in enumerate(sequences):
            if w>0:
                i = self.word_index.keys()[self.word_index.values().index(w)]
                #print i
                vect.append(i)


        return vect
        
    def matrix_to_sequences(self, matrix):

        seq = np.argmax(matrix, axis = -1)
        

        return seq  
        
    def matrix_to_text(self, matrix):
        seq=self.matrix_to_sequences(matrix)
        text=self.sequences_to_text(seq)

        return text        
        
def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post', truncating='post', value=0.):
    """
        Pad each sequence to the same length: 
        the length of the longuest sequence.
        If maxlen is provided, any sequence longer
        than maxlen is truncated to maxlen. Truncation happens off either the beginning (default) or
        the end of the sequence.
        Supports post-padding and pre-padding (default).
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
        print maxlen

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


if __name__ == "__main__":
    n_batch=9000
    text=[]
    with open("OpenSubtitles2012.en-ko.en") as f:
        for line in f:
            text.append(line)
    input=Tokenizer(500)
    input.fit_on_texts(text)
    train_x=input.texts_to_matrix(text,"binary",n_batch)
    
    a=input.matrix_to_text(train_x[10])
    
    '''
    text=[]
    with open("news-commentary-v9.fr-en.fr") as f:
        for line in f:
            text.append(line)

    output=Tokenizer(500)
    output.fit_on_texts(text)
    train_y=output.texts_to_sequences(text,n_batch)
    train_y=pad_sequences(train_y)
    '''
    

                



    