import rampy as rp
import numpy as np

def residual_l(pars, x, data=None, eps=None): #Function definition
    # unpack parameters, extract .value attribute for each parameter
    n = len(pars)//3

    la = ['a'+str(i) for i in range(1,n+1)]
    lf = ['f'+str(i) for i in range(1,n+1)]
    ll = ['l'+str(i) for i in range(1,n+1)]
            
    a = [pars[i].value for i in la]
    f = [pars[i].value for i in lf]
    l = [pars[i].value for i in ll]
            
    # Using the Gaussian model function from rampy
            
    peaks = [rp.lorentzian(x,a[i],f[i],l[i]) for i in range(n)]


    model = np.zeros(peaks[0].shape)
    for p in peaks:
        model += p
            
    if data is None: # if we don't have data, the function only returns the direct calculation
        return model, peaks
    if eps is None: # without errors, no ponderation
        return (model - data)
    return (model - data)/eps # with errors, the difference is ponderated

def residual_g(pars, x, data=None, eps=None): #Function definition
    # unpack parameters, extract .value attribute for each parameter
    n = len(pars)//3

    la = ['a'+str(i) for i in range(1,n+1)]
    lf = ['f'+str(i) for i in range(1,n+1)]
    ll = ['l'+str(i) for i in range(1,n+1)]
            
    a = [pars[i].value for i in la]
    f = [pars[i].value for i in lf]
    l = [pars[i].value for i in ll]
            
    # Using the Gaussian model function from rampy
            
    peaks = [rp.gaussian(x,a[i],f[i],l[i]) for i in range(n)]

    model = np.zeros(peaks[0].shape)
    for p in peaks:
        model += p
            
    if data is None: # if we don't have data, the function only returns the direct calculation
        return model, peaks
    if eps is None: # without errors, no ponderation
        return (model - data)
    return (model - data)/eps # with errors, the difference is ponderated

