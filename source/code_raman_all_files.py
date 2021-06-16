import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rampy as rp
import os

two_peaks = False
results_folder = 'results_all_peaks'

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

arqs = os.listdir('../raman_dados/')
Data = []
amostras = []
for ind_arq in range(len(arqs)):
    file = arqs[ind_arq]
    if file[-3:] == 'txt':

        R = pd.read_csv('../raman_dados/'+file,header = None, sep = '\t')
        R = R.values

        roi = np.array([[260,800],[1900,2100]])
        x, y = R[:,0], R[:,1]

        # calculating the baselines
        ycalc_poly, base_poly = rp.baseline(x,y,roi,'poly',polynomial_order=3 )

        import lmfit
        from lmfit.models import GaussianModel

        inputsp = R
        y_corr = ycalc_poly
        # signal selection
        lb = 750 # The lower boundary of interest
        hb = 2000 # The upper boundary of interest
        x = inputsp[:,0]
        x_fit = x[np.where((x > lb)&(x < hb))]
        y_fit = y_corr[np.where((x > lb)&(x < hb))]
        ese0 = np.sqrt(abs(y_fit[:,0]))/abs(y_fit[:,0]) # the relative errors after baseline subtraction
        YM = np.amax(y_fit[:,0])*10
        y_fit[:,0] = y_fit[:,0]/YM # normalise spectra to maximum intensity, easier to handle 


        params = lmfit.Parameters()
        #               (Name,  Value,  Vary,   Min,  Max,  Expr)
        params.add_many(('a1',   2.4,   True,  0,      None,  None),
                ('f1',   1245,   True, 1100,    1300,  None),
                ('l1',   26,   True,  0,      None,  None),
                ('a2',   3.5,   True,  0,      None,  None),
                ('f2',   1350,  True, 1300,   1400,  None),
                ('l2',   39,   True,  0,   None,  None),  
                ('a3',   8.5,    True,    0,      None,  None),
                ('f3',   1510,  True, 1400,   1600,  None),
                ('l3',   31,   True,  0,   None,  None),
                ('a4',   8.5,    True,    0,      None,  None),
                ('f4',   1605,  True, 1500,   1700,  None),
                ('l4',   31,   True,  0,   None,  None))

        n = len(params)//3
        lf = ['f'+str(i) for i in range(1,n+1)]
        for i in lf:
            params[i].vary = False

        algo = 'cg'  
            
        result = lmfit.minimize(residual_l, params, method = algo, args=(x_fit, y_fit[:,0])) # fit data with  nelder model from scipy


        # we release the positions but contrain the FWMH and amplitude of all peaks 
        n = len(params)//3
        lf = ['f'+str(i) for i in range(1,n+1)]
        if two_peaks:
            freq_free = lf[1::2]
        else:
            freq_free = lf
        for i in freq_free:
            params[i].vary = True

        #result2 = lmfit.minimize(residual_l, params, method = algo, args=(x_fit, y_fit[:,0])) # fit data with leastsq model from scipy
        result2 = lmfit.minimize(residual_g, params, method = algo, args=(x_fit, y_fit[:,0])) # fit data with leastsq model from scipy

        model = lmfit.fit_report(result2.params)
        yout, peaks = residual_g(result2.params,x_fit) # the different peaks
        r_p = result2.params.valuesdict()

        plt.figure()
        plt.plot(x_fit,y_fit,'k-')
        for p in peaks:
            plt.plot(x_fit,p,'b-')
        plt.plot(x_fit,yout,'r-')
            
        plt.xlim(lb,hb)
        plt.xlabel("Raman shift, cm$^{-1}$", fontsize = 14)
        plt.ylabel("Normalized intensity, a. u.", fontsize = 14)
        plt.title("Fitted peaks",fontsize = 14,fontweight = "bold")
        for i in range(n):
            plt.annotate('{:.2f}'.format(r_p['f'+str(i+1)]),(r_p['f'+str(i+1)],r_p['a'+str(i+1)]),color='red')
        plt.savefig('../'+results_folder+'/'+file[:-3]+'png')

        res = dict(result2.params.valuesdict())
        la = ['a'+str(i) for i in range(1,n+1)]
        for i in la:
            res[i] = res[i]*YM

        Data += [res]
        amostras += [arqs[ind_arq][:-4]]

a1 = [d['a1'] for d in Data]
a2 = [d['a2'] for d in Data]
a3 = [d['a3'] for d in Data]
a4 = [d['a4'] for d in Data]
f1 = [d['f1'] for d in Data]
f2 = [d['f2'] for d in Data]
f3 = [d['f3'] for d in Data]
f4 = [d['f4'] for d in Data]
l1 = [d['l1'] for d in Data]
l2 = [d['l2'] for d in Data]
l3 = [d['l3'] for d in Data]
l4 = [d['l4'] for d in Data]


d = {'a1':a1,'a2':a2,'a3':a3,'a4':a4,'f1':f1,'f2':f2,'f3':f3,'f4':f4,'l1':l1,'l2':l2,'l3':l3,'l4':l4,'arquivo':amostras}
print(d)
tabela = pd.DataFrame(d)
tabela.to_csv('../'+results_folder+'/picos_ajustados.csv')

from sklearn.decomposition import PCA
dd = tabela.iloc[:,:-1].values
X = dd
pca = PCA(n_components = 2)
pc = pca.fit_transform(X)
pcvars = pca.explained_variance_ratio_[:2]
fig = plt.figure() 
plt.plot(pc[:,0],pc[:,1],'o')
for i in range(pc.shape[0]):
    
    x = pc[i,0]
    y = pc[i,1]

    label = str(amostras[i])

    plt.annotate(label, # this is the text
                    (x,y), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0,3), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center        plt.xlabel(f'PC1 ({pcvars[0]:.3f}%)')
plt.xlabel(f'PC1 ({100*pcvars[0]:.3f}%)')        
plt.ylabel(f'PC2 ({100*pcvars[1]:.3f}%)')
plt.title('PCA')
plt.savefig('../'+results_folder+'/pca_raman.png')