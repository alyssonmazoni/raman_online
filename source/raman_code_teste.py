import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rampy as rp
import os


R = pd.read_csv('./raman_dados/Denezine_MP3710_Aj.txt',header = None, sep = '\t')
R = R.values

plt.plot(R[:,0],R[:,1])

