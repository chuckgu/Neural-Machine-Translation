from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np
from initializations import shared_zeros
from collections import OrderedDict
from six.moves import zip


<<<<<<< HEAD
def SGD(params,cost,mom,lr):
=======
def SGD(params,cost,update,mom,lr):
>>>>>>> c6bc144111175998c997e5e6e9d44519b73e732d

    gparams = []

    for param in params:
        gparams.append(T.grad(cost, param))
    
    # zip just concatenate two lists
    updates = OrderedDict()
    for param, gparam in zip(params, gparams):
<<<<<<< HEAD
        weight_update = theano.shared(param.get_value(borrow = True) * 0.)
=======
        weight_update = update[param]
>>>>>>> c6bc144111175998c997e5e6e9d44519b73e732d
        upd = mom*weight_update - lr * gparam
        updates[weight_update] = upd
        updates[param] = param + upd

    return updates


<<<<<<< HEAD
def RMSprop(params,cost,mom,lr,rho=0.9, epsilon=1e-6):
=======
def RMSprop(params,cost,update,mom,lr,rho=0.9, epsilon=1e-6):
>>>>>>> c6bc144111175998c997e5e6e9d44519b73e732d

    gparams = []
    for param in params:
        gparams.append(T.grad(cost, param))
    
    # zip just concatenate two lists
    updates = OrderedDict()
    for param, gparam in zip(params, gparams):
        acc = theano.shared(param.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * gparam** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        gparam = gparam / gradient_scaling
        
        upd = - lr * gparam
        updates[acc]=acc_new
        updates[param] = param + upd

    return updates    


<<<<<<< HEAD
def Adagrad(params,cost,mom,lr,epsilon=1e-6):
=======
def Adagrad(params,cost,update,mom,lr,epsilon=1e-6):
>>>>>>> c6bc144111175998c997e5e6e9d44519b73e732d

    gparams = []
    for param in params:
        gparams.append(T.grad(cost, param))
    
    # zip just concatenate two lists
    updates = OrderedDict()
    for param, gparam in zip(params, gparams):
        acc = theano.shared(param.get_value() * 0.)
        acc_new = acc +gparam** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        gparam = gparam / gradient_scaling
        
        upd = - lr * gparam
        updates[acc]=acc_new
        updates[param] = param + upd

    return updates 


<<<<<<< HEAD
def Adadelta(params,cost,mom,lr,rho=0.95, epsilon=1e-6):
=======
def Adadelta(params,cost,update,mom,lr,rho=0.95, epsilon=1e-6):
>>>>>>> c6bc144111175998c997e5e6e9d44519b73e732d

    gparams = []
    for param in params:
        gparams.append(T.grad(cost, param))
    
    # zip just concatenate two lists
    updates = OrderedDict()
    for param, gparam in zip(params, gparams):
        acc = theano.shared(param.get_value() * 0.)
        d_acc = theano.shared(param.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * gparam** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        update = gparam * T.sqrt(d_acc + epsilon) / gradient_scaling
        
        upd = - lr * update
        
        new_d_acc = rho * d_acc + (1 - rho) * update ** 2
        
        updates[acc]=acc_new
        updates[d_acc]=new_d_acc
        updates[param] = param + upd

    return updates    








