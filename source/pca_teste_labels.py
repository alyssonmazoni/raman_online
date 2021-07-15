import pandas as pd
import matplotlib.pyplot as plt

results_folder = 'results_two_peaks'

table = pd.read_csv('../'+results_folder+'/picos_ajustados.csv')
labels_exp = pd.read_excel('../raman_dados/Tabela_Raman.xlsx')



from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

dd = table.iloc[:,:-1].values
amostras = table.iloc[:,-1].values
X = StandardScaler().fit_transform(dd)
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
                    ha='center') # horizontal alignment can be left, right or center
plt.xlabel(f'PC1 ({100*pcvars[0]:.3f}%)')        
plt.ylabel(f'PC2 ({100*pcvars[1]:.3f}%)')
plt.title('PCA')
plt.savefig('../'+results_folder+'/pca_raman.png')