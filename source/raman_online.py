import streamlit as st
import pandas as pd
import numpy as np
import rampy as rp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import lmfit
from lmfit.models import GaussianModel

import residuals as resid

two_peaks = True

st.title('Raman spectroscopy processing')

files = st.file_uploader('Files to be processed',accept_multiple_files=True)

@st.cache(suppress_st_warning=True)
def process_files(files,base_start0,base_start1,base_end0,base_end1,lb,hb):
    Data = []
    amostras = []
    for ind_f,f in enumerate(files):
        R = pd.read_csv(f,header = None, sep = '\t')
        R = R.values
        roi = np.array([[base_start0,base_start1],[base_end0,base_end1]])
        x, y = R[:,0], R[:,1]

        # calculating the baselines
        ycalc_poly, base_poly = rp.baseline(x,y,roi,'poly',polynomial_order=3)

        inputsp = R
        y_corr = ycalc_poly

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
                ('f3',   1510,  True, 1400,   1550,  None),
                ('l3',   31,   True,  0,   None,  None),
                ('a4',   8.5,    True,    0,      None,  None),
                ('f4',   1605,  True, 1600,   1700,  None),
                ('l4',   31,   True,  0,   None,  None))
        
        
        n = len(params)//3
        lf = ['f'+str(i) for i in range(1,n+1)]
        for i in lf:
            params[i].vary = False

        algo = 'cg'  
        result = lmfit.minimize(resid.residual_g, params, method = algo, args=(x_fit, y_fit[:,0])) # fit data with  nelder model from scipy


        # we release the positions but contrain the FWMH and amplitude of all peaks 
        n = len(params)//3
        lf = ['f'+str(i) for i in range(1,n+1)]
        if two_peaks:
            freq_free = lf[1::2]
        else:
            freq_free = lf
        for i in freq_free:
            params[i].vary = True

        result2 = lmfit.minimize(resid.residual_g, params, method = algo, args=(x_fit, y_fit[:,0])) # fit data with leastsq model from scipy

        model = lmfit.fit_report(result2.params)
        yout, peaks = resid.residual_g(result2.params,x_fit) # the different peaks
        r_p = result2.params.valuesdict()

        fig = plt.figure()
        plt.plot(x_fit,y_fit,'k-')
        for p in peaks:
            plt.plot(x_fit,p,'b-')
        plt.plot(x_fit,yout,'r-')
            
        plt.xlim(lb,hb)
        plt.xlabel("Raman shift, cm$^{-1}$", fontsize = 14)
        plt.ylabel("Normalized intensity, a. u.", fontsize = 14)
        plt.title("Fitted peaks: "+f.name[:-4],fontsize = 14,fontweight = "bold")
        for i in range(n):
            plt.annotate('{:.2f}'.format(r_p['f'+str(i+1)]),(r_p['f'+str(i+1)],r_p['a'+str(i+1)]),color='red')

        st.pyplot(fig) 

        res = dict(result2.params.valuesdict())
        la = ['a'+str(i) for i in range(1,n+1)]
        for i in la:
            res[i] = res[i]*YM

        Data += [res]
        amostras += [f.name[:-4]]
        
    return Data, amostras

    
    
if len(files)>0:

    st.subheader('Baseline calculation')
    st.subheader('Start region of the baseline')
    base_start0 = st.number_input('Begining of start region',help='Region at the begining of data whereto estimate baseline.',min_value=0,value = 200,key=-1)
    base_start1 = st.number_input('Ending of start region',help='Region at the begining of data whereto estimate baseline.',min_value=base_start0,value=800,key=-2)
    st.subheader('End region of the baseline')
    base_end0 = st.number_input('Begining of end region of the baseline',help='Region at the end of data whereto estimate baseline.',min_value=base_start1,value=1900,key=-3)
    base_end1 = st.number_input('Ending of end region of the baseline',help='Region at the end of data whereto estimate baseline.',min_value=base_end0,value=2100,key=-4)
    st.subheader('Region of interest')
    # signal selection
    lb = st.number_input('Lower bound of interest',min_value=0,value=800,key=0) # The lower boundary of interest
    hb = st.number_input('Upper bound of interest',min_value=1000,value=1800,key=1) # The upper boundary of interest

    Data,amostras = process_files(files,base_start0,base_start1,base_end0,base_end1,lb,hb)

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

    tabela = pd.DataFrame(d)
    st.subheader('Results: amplitude, Raman shift, width, sample name')
    st.write(tabela)    

    st.subheader('PCA')
    dd = tabela.iloc[:,:-1].values
    X = dd
    pca = PCA(n_components = 2)
    pc = pca.fit_transform(X)
    pcvars = pca.explained_variance_ratio_[:2]
    fig = plt.figure() 
    plt.plot(pc[:,0],pc[:,1],'o')

    plt.xlabel(f'PC1 ({100*pcvars[0]:.3f}%)')        
    plt.ylabel(f'PC2 ({100*pcvars[1]:.3f}%)')
    plt.title('PCA')

    lp = st.checkbox('Label points?',value=True)

    if lp:
        sel_label = st.selectbox("Select labels for the points",tabela.columns,index=12)
        labels = tabela[sel_label]
        
        for i in range(pc.shape[0]):
    
            x = pc[i,0]
            y = pc[i,1]

            label = str(labels[i])

            plt.annotate(label, # this is the text
                        (x,y), # this is the point to label
                        textcoords="offset points", # how to position the text
                        xytext=(0,3), # distance from text to points (x,y)
                        ha='center') # horizontal alignment can be left, right or center 
                        
    st.pyplot(fig)
