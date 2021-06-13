# -*- coding: utf-8 -*-
"""
Created on Sun May 30 12:26:23 2021

@author: SampathKumar.AP
"""

# packages & modules
import pandas as pd
from random import sample
import numpy as np
from matplotlib import pyplot as plt

# data
data = pd.read_excel(r'D:\Sampathkumar.AP\Desktop\Work\Blog stuff\Prob Distributions- Simple workings\clt.xlsx')

# inputs
no_samples = 2000
sample_size = 1000

# main
freq_plot = data.groupby('no_children').agg({'no_children':'count'})
freq_plot.columns = ['count']
freq_plot.reset_index(inplace = True)
plt.bar(freq_plot['no_children'], freq_plot['count'])

sampling_dist_mean = [data.iloc[sample(range(len(data)), sample_size), 1].mean() for i in range(no_samples)]

# np.linspace(start=min(sampling_dist_mean), stop=max(sampling_dist_mean),num=11, endpoint=True)
plt.hist(x=sampling_dist_mean, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
