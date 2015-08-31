import theano
import theano.tensor as T
import numpy as np
import copy
import logging
import os
import datetime
import cPickle as pickle
import Loss
from collections import OrderedDict
from initializations import glorot_uniform,zero,alloc_zeros_matrix,norm_weight,glorot_normal
from utils import Progbar
from optimizers import SGD,RMSprop,Adagrad,Adadelta


logger = logging.getLogger(__name__)

mode = theano.Mode(linker='cvm') #the runtime algo to execute the code is in c

def ndim_tensor(ndim):
    if ndim == 2:
        return T.matrix()
    elif ndim == 3:
        return T.tensor3()
    elif ndim == 4:
        return T.tensor4()
    return T.matrix()

class ENC_DEC(object):
    
    def __init__(self,n_in,n_hidden,n_decoder,n_out,
<<<<<<< HEAD
                 n_epochs=400,n_chapter=100,n_batch=16,maxlen=20,n_words_x=10000,n_words_y=10000,dim_word=100,
=======
                 n_epochs=400,n_chapter=100,n_batch=16,maxlen=20,n_words=10000,dim_word=100,
>>>>>>> c6bc144111175998c997e5e6e9d44519b73e732d
                 momentum_switchover=5,lr=0.001,learning_rate_decay=0.999,snapshot=100,sample_Freq=100,val_Freq=100,L1_reg=0,L2_reg=0):
        
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.n_decoder=int(n_decoder)
        self.n_out=int(n_out)
        
        self.n_batch=int(n_batch)
        
        if n_chapter is not None:self.n_chapter=int(n_chapter)
        else:self.n_chapter=None
        
        self.n_epochs=n_epochs
        self.maxlen= int(maxlen)   
        self.dim_word=dim_word
        self.n_words_x=n_words_x
        self.n_words_y=n_words_y
        
        self.x = T.matrix(name = 'x', dtype = 'int32')
        self.y = T.matrix(name = 'y', dtype = 'int32')
        
        self.x_mask = T.matrix(name = 'x_mask', dtype = 'float32')
        self.y_mask = T.matrix(name = 'y_mask', dtype = 'float32')
        
        self.x_emb = T.tensor3(name = 'x', dtype = 'float32')
        self.y_emb = T.tensor3(name = 'y', dtype = 'float32')        
        
        self.W_hy = glorot_uniform((self.n_out,self.n_words_y))
        self.b_hy = zero((self.n_words_y,))
        
        self.W_hi = glorot_uniform((self.n_hidden,self.n_decoder))
        self.b_hi = zero((n_decoder,))
        
        self.Wemb=glorot_normal((self.n_words_x,self.dim_word))
        #self.Wemb_dec=glorot_normal((self.n_words_y,self.dim_word))
         
        self.layers = []
        self.decoder=[]
        self.params=[]
        self.errors=[]
        
        #self.updates = {}
        

        self.initial_momentum=0.5
        self.final_momentum=0.9
        self.lr=float(lr)
        self.momentum_switchover=int(momentum_switchover)
        self.learning_rate_decay=learning_rate_decay
        
        self.snapshot=int(snapshot)
        self.sample_Freq=int(sample_Freq)
        self.val_Freq=int(val_Freq)
       
        self.L1_reg=L1_reg
        self.L2_reg=L2_reg    
        self.L1= 0
        self.L2_sqr= 0
        
        
        ## word embedding 

        self.x_emb=self.Wemb[T.cast(self.x.flatten(),'int32')].reshape((self.x.shape[0], self.x.shape[1], self.dim_word))
        
        self.y_emb=self.Wemb[T.cast(self.y.flatten(),'int32')].reshape((self.y.shape[0], self.y.shape[1], self.dim_word))
        
        
        
    def add(self,layer): 
  
        self.layers.append(layer)
    
        if len(self.layers) > 1:
            self.layers[-1].set_previous(self.layers[-2])
        else:
            self.layers[0].set_input(self.x_emb)
            self.layers[0].set_mask(self.x_mask)
  
        self.params+=layer.params
        self.L1 += layer.L1
        self.L2_sqr += layer.L2_sqr
    

    def set_params(self,**params):
        return
    
    def __getstate__(self):
        """ Return state sequence."""
        params = self.params  # parameters set in constructor
        weights = [p.get_value() for p in self.params]
        lr=self.lr
        error=self.errors
        state = (params, weights,lr,error)
        return state

    def _set_weights(self, weights):
        """ Set fittable parameters from weights sequence.
        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        i = iter(weights)

        for param in self.params:
            param.set_value(i.next())

    def __setstate__(self, state):
        """ Set parameters from state sequence.
        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        params, weights, lr,error = state
        #self.set_params(**params)
        #self.ready()
        self._set_weights(weights)
        self.lr=lr
        self.errors=error

    def save(self, fpath='.', fname=None):
        """ Save a pickled representation of Model state. """
        fpathstart, fpathext = os.path.splitext(fpath)
        if fpathext == '.pkl':
            # User supplied an absolute path to a pickle file
            fpath, fname = os.path.split(fpath)

        elif fname is None:
            # Generate filename based on date
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
            class_name = self.__class__.__name__
            fname = '%s.%s.pkl' % (class_name, date_str)

        fabspath = os.path.join(fpath, fname)

        logger.info("Saving to %s ..." % fabspath)
        print("Saving to %s ..." % fabspath)
        file = open(fabspath, 'wb')
        state = self.__getstate__()
        pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

    def load(self, path):
        """ Load model parameters from path. """
        logger.info("Loading from %s ..." % path)
        print("Loading from %s ..." % path)
        file = open(path, 'rb')
        state = pickle.load(file)
        self.__setstate__(state)
        file.close()       
        
    
    def get_output(self):
        ctx=self.layers[-1].get_input()
        ctx_mean = (ctx * self.x_mask[:,:,None]).sum(0) / self.x_mask.sum(0)[:,None]
        init_state=T.tanh(T.dot(ctx_mean, self.W_hi) + self.b_hi)
        
        return self.layers[-1].get_output(self.y_emb,self.y_mask,init_state)
<<<<<<< HEAD

=======
>>>>>>> c6bc144111175998c997e5e6e9d44519b73e732d
        
    def get_sample(self,y,h):
        
       

        ctx=self.layers[-1].get_input()
        ctx_mean = (ctx * self.x_mask[:,:,None]).sum(0) / self.x_mask.sum(0)[:,None]
        
        h = T.switch(h[0] < 0, 
                        T.tanh(T.dot(ctx_mean, self.W_hi) + self.b_hi), 
                        h) 
        
        h,logit=self.layers[-1].get_sample(y,h)
        y_gen = T.dot(logit, self.Wemb.T)
            
        p_y_given_x_gen=T.nnet.softmax(y_gen)
            
        return h,logit,p_y_given_x_gen  

    def set_input(self):
        for l in self.layers:
            if hasattr(l, 'input'):
                ndim = l.input.ndim
                self.layers[0].input = ndim_tensor(ndim)
                break

    def get_input(self, train=False):
        if not hasattr(self.layers[0], 'input'):
            self.set_input()
        return self.layers[0].get_input()  
        
       

        
    def build(self):      
     
        ### set up parameters
    
        self.params+=[self.W_hi, self.b_hi, self.Wemb]
        '''
        for param in self.params:
            self.updates[param] = theano.shared(
                                      value = np.zeros(
                                                  param.get_value(
                                                      borrow = True).shape,
                                                      dtype = theano.config.floatX),
                                      name = 'updates')        
        '''
        ### set up regularizer                               
   
        self.L1 += T.sum(abs(self.W_hy))    
        self.L2_sqr += T.sum(self.W_hy**2)
                                                                  
        ### fianl prediction formular
                                             
               
        self.y_pred = T.dot(self.get_output(), self.Wemb.T)
        
        y_p = self.y_pred
        y_p_m = T.reshape(y_p, (y_p.shape[0] * y_p.shape[1], -1))
        y_p_s = T.nnet.softmax(y_p_m)
        self.p_y_given_x = T.reshape(y_p_s, y_p.shape)
                
        
        self.loss = lambda y,y_mask: Loss.nll_multiclass(self.p_y_given_x,y,y_mask)
        

    def train(self,X_train,X_mask,Y_train,Y_mask,input,output,verbose,optimizer):

        train_set_x = theano.shared(np.asarray(X_train, dtype='int32'), borrow=True)
        train_set_y = theano.shared(np.asarray(Y_train, dtype='int32'), borrow=True)
        
        mask_set_x = theano.shared(np.asarray(X_mask, dtype='float32'), borrow=True)
        mask_set_y = theano.shared(np.asarray(Y_mask, dtype='float32'), borrow=True)
        

        index = T.lscalar('index')    # index to a case    
        lr = T.scalar('lr', dtype = theano.config.floatX)
        mom = T.scalar('mom', dtype = theano.config.floatX)  # momentum
        n_ex = T.lscalar('n_ex')
        sindex = T.lscalar('sindex')    # index to a case
        

        ### batch
        
        batch_start=index*self.n_batch
        batch_stop=T.minimum(n_ex,(index+1)*self.n_batch)
        
        
        effective_batch_size = batch_stop - batch_start

        get_batch_size = theano.function(inputs=[index, n_ex],
                                          outputs=effective_batch_size)
                                          
        
        
        cost = self.loss(self.y,self.y_mask) +self.L1_reg * self.L1
<<<<<<< HEAD
  
        updates=eval(optimizer)(self.params,cost,mom,lr)    
        
        
        
=======
        '''
        gparams = []
        for param in self.params:
            gparams.append(T.grad(cost, param))

        # zip just concatenate two lists
        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            weight_update = self.updates[param]
            upd = mom*weight_update - lr * gparam
            updates[weight_update] = upd
            updates[param] = param + upd
        '''    
        updates=eval(optimizer)(self.params,cost,self.updates,mom,lr)    
>>>>>>> c6bc144111175998c997e5e6e9d44519b73e732d
        '''    
        compute_val_error = theano.function(inputs = [index,n_ex ],
                                              outputs = self.loss(self.y,self.y_mask),
                                              givens = {
                                                  self.x: train_set_x[:,batch_start:batch_stop],
                                                  self.y: train_set_y[:,batch_start:batch_stop],
                                                  self.x_mask: mask_set_x[:,batch_start:batch_stop],
                                                  self.y_mask: mask_set_y[:,batch_start:batch_stop]  
                                                    },
                                              mode = mode)    
        '''
        train_model =theano.function(inputs = [index, lr, mom,n_ex],
                                      outputs = [cost,self.loss(self.y,self.y_mask)],
                                      updates = updates,
                                      givens = {
                                            self.x: train_set_x[:,batch_start:batch_stop],
                                            self.y: train_set_y[:,batch_start:batch_stop],
                                            self.x_mask: mask_set_x[:,batch_start:batch_stop],
                                            self.y_mask: mask_set_y[:,batch_start:batch_stop]  
                                                    },
<<<<<<< HEAD
                                      mode = mode,
                                      on_unused_input='ignore')
=======
                                      mode = mode)
>>>>>>> c6bc144111175998c997e5e6e9d44519b73e732d

        ###############
        # TRAIN MODEL #
        ###############
        print 'Training model ...'
        epoch = 0
        n_train = train_set_x.get_value(borrow = True).shape[1]
        n_train_batches = int(np.ceil(1.0 * n_train / self.n_batch))
        
        if optimizer is not 'SGD': self.learning_rate_decay=1
        
        while (epoch < self.n_epochs):
            epoch = epoch + 1
            if verbose==1: 
                progbar=Progbar(n_train_batches)
            train_losses=[]
            train_batch_sizes=[]
            for idx in xrange(n_train_batches):
            
                effective_momentum = self.final_momentum \
<<<<<<< HEAD
                                     if (epoch+len(self.errors)) > self.momentum_switchover \
=======
                                     if epoch > self.momentum_switchover \
>>>>>>> c6bc144111175998c997e5e6e9d44519b73e732d
                                     else self.initial_momentum
                cost = train_model(idx,
                                   self.lr,
                                   effective_momentum,n_train)
                                   
                train_losses.append(cost[1]) 
                train_batch_sizes.append(get_batch_size(idx, n_train))    
                          
                if verbose==1: progbar.update(idx+1)                            
                            
            this_train_loss = np.average(train_losses,
                                         weights=train_batch_sizes)      
                       
            self.errors.append(this_train_loss)
           
           
           
            print('epoch %i, train loss %f ''lr: %f' % \
                  (epoch, this_train_loss, self.lr))
                  
                  
            ### autimatically saving snapshot ..
            if np.mod(epoch,self.snapshot)==0:
                if epoch is not n_train_batches: self.save()
            
            ### generating sample.. 
            if np.mod(epoch,self.sample_Freq)==0:
                print 'Generating a sample...'               
                
                i=np.random.randint(1,n_train)
                
                test=X_train[:,i]
<<<<<<< HEAD

                truth=Y_train[:,i]                
                
                guess =self.gen_sample(test,X_mask[:,i])
                
                print 'Input: ',' '.join(input.sequences_to_text(test))
    
                print 'Truth: ',' '.join(output.sequences_to_text(truth))
                
                print 'Sample: ',' '.join(output.sequences_to_text(guess[1]))
             
            '''
            # compute loss on validation set
            if np.mod(epoch,self.val_Freq)==0:

                val_losses = [compute_val_error(i, n_train)
                                for i in xrange(n_train_batches)]
                val_batch_sizes = [get_batch_size(i, n_train)
                                     for i in xrange(n_train_batches)]
                this_val_loss = np.average(val_losses,
                                         weights=val_batch_sizes)                     
            '''     
                
            self.lr *= self.learning_rate_decay
                    
                          
   

                                      
    def gen_sample(self,X_test,X_mask,stochastic=True,k=3):
        
        ### define symbollic structure
        next_y=T.matrix()
        next_h=T.matrix()
        
        get_sample = theano.function(inputs = [self.x,self.x_mask,next_y,next_h],
                                             outputs = self.get_sample(next_y,next_h),
                                             mode = mode,
                                             on_unused_input='ignore')        
        r=T.lscalar()                                     
        get_vector = theano.function(inputs = [r,],
                                     outputs = self.Wemb[r],
                                     mode = mode,
                                     on_unused_input='ignore')  
                                             
        X_test=np.asarray(X_test[:,None],dtype='int32')
        X_mask=np.asarray(X_mask[:,None],dtype='float32')

=======

                truth=Y_train[:,i]                
                
                guess =self.gen_sample(test,X_mask[:,i])
                
                print 'Input: ',' '.join(input.sequences_to_text(test))
    
                print 'Truth: ',' '.join(output.sequences_to_text(truth))
                
                print 'Sample: ',' '.join(output.sequences_to_text(guess[1]))
             
            '''
            # compute loss on validation set
            if np.mod(epoch,self.val_Freq)==0:

                val_losses = [compute_val_error(i, n_train)
                                for i in xrange(n_train_batches)]
                val_batch_sizes = [get_batch_size(i, n_train)
                                     for i in xrange(n_train_batches)]
                this_val_loss = np.average(val_losses,
                                         weights=val_batch_sizes)                     
            '''     
                
            self.lr *= self.learning_rate_decay
                    
                          
   

                                      
    def gen_sample(self,X_test,X_mask,stochastic=True):
        X_test=np.asarray(X_test[:,None],dtype='int32')
        X_mask=np.asarray(X_mask[:,None],dtype='float32')

        
        
>>>>>>> c6bc144111175998c997e5e6e9d44519b73e732d
        sample=[]
        sample_proba=[]
        
        sample_score = []
        
        live_k = 1
        dead_k = 0
        
        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k).astype('float32')
        hyp_states = []

<<<<<<< HEAD
        next_w=np.zeros((1,self.n_out)).astype('float32') 
        h_w=-1*np.ones((1,self.n_decoder)).astype('float32')
        

        

=======
        next_w=T.zeros((1,self.n_out))     
        ctx=self.layers[-1].get_input()
        ctx_mean = (ctx * X_mask[:,:,None]).sum(0) / X_mask.sum(0)[:,None]
        h_w=T.tanh(T.dot(ctx_mean, self.W_hi) + self.b_hi)
        


        for i in xrange(self.maxlen):
            
            h_w,logit,c=self.get_sample(next_w,h_w)
>>>>>>> c6bc144111175998c997e5e6e9d44519b73e732d

        for i in xrange(self.maxlen):
            
            h_w,logit,p_y_given_x_gen=get_sample(X_test,X_mask,next_w,h_w)
            sample_proba.append(p_y_given_x_gen.flatten())
            
            if stochastic: ### stochastic sampling
                
                result = np.argmax(p_y_given_x_gen, axis = -1)[0] 
                

                sample.append(result) 
                sample_score.append(c)
                
<<<<<<< HEAD
                w=get_vector(result)
                
                next_w=np.asarray(w.reshape((1,self.n_out))).astype('float32')

             
            else:   
                p_y_given_x_gen=np.array(p_y_given_x_gen).astype('float32')

                #print p_y_given_x_gen
                cand_scores = hyp_scores[:,None] - np.log(p_y_given_x_gen.flatten())
=======
                next_w=self.Wemb_dec[result]

        '''        
            else:    #Todo : implement Beam Search Algorithm here
            
                cand_scores = hyp_scores[:,None] - np.log(p_y_given_x_gen)
>>>>>>> c6bc144111175998c997e5e6e9d44519b73e732d
                cand_flat = cand_scores.flatten()
                ranks_flat = cand_flat.argsort()[:(k-dead_k)]
                
                voc_size = p_y_given_x_gen.shape[1]
                trans_indices = ranks_flat / voc_size
                word_indices = ranks_flat % voc_size
                costs = cand_flat[ranks_flat]
    
                new_hyp_samples = []
                new_hyp_scores = np.zeros(k-dead_k).astype('float32')
               # new_hyp_states = []

                
                for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                    
                    new_hyp_samples.append(hyp_samples[ti]+[wi])
                    new_hyp_scores[idx] = copy.copy(costs[ti])
                   # new_hyp_states.append(copy.copy(result[ti]))
                
                # check the finished samples
                new_live_k = 0
                hyp_samples = []
                hyp_scores = []
                #hyp_states = []
    
                for idx in xrange(len(new_hyp_samples)):
                    if new_hyp_samples[idx][-1] == 0:
                        sample.append(new_hyp_samples[idx])
                        sample_score.append(new_hyp_scores[idx])
                        dead_k += 1
                    else:
                        new_live_k += 1
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        #hyp_states.append(new_hyp_states[idx])
                hyp_scores = np.array(hyp_scores)
                live_k = new_live_k
    
                if new_live_k < 1:
                    break
                if dead_k >= k:
                    break
    
                next_w = np.array([w[-1] for w in hyp_samples])
                w=get_vector(next_w[0])
                next_w=np.asarray(w.reshape((1,self.n_out))).astype('float32')
                #next_state = np.array(hyp_states)
        
        if not stochastic:
        # dump every remaining one
<<<<<<< HEAD
            if live_k > 0:
                for idx in xrange(live_k):
                    sample.append(hyp_samples[idx])
                    sample_score.append(hyp_scores[idx])
            sample=sample[np.argmin(sample_score)]        
            print sample        

        return sample_proba,sample
   
=======
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])
        '''
        ## compile theano graph
        predict_proba = theano.function(inputs = [self.x,self.x_mask],
                                             outputs = sample_proba,
                                             mode = mode)
                                             
        predict = theano.function(inputs = [self.x,self.x_mask],
                                       outputs = sample, # y-out is calculated by applying argmax
                                       mode = mode)          
        '''                               
        predict_etc = theano.function(inputs = [self.x,self.x_mask],
                                       outputs = sample_score, # y-out is calculated by applying argmax
                                       mode = mode,
                                       on_unused_input='ignore')            
        '''
                                       
        return  predict_proba(X_test,X_mask),predict(X_test,X_mask)
        
>>>>>>> c6bc144111175998c997e5e6e9d44519b73e732d
        