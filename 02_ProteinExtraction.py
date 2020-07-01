# -*- coding: utf-8 -*-
"""
Created on Sat May  2 22:04:56 2020

@author: Marc Jerrone Castro
"""

import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

raw = pd.read_csv('input_data/biogrid.ppi.data.txt',delimiter="\t")
df = pd.DataFrame()
df['InteractorA'] = raw['SWISS-PROT Accessions Interactor A']
df['InteractorB'] = raw['SWISS-PROT Accessions Interactor B']
df['InteractorA'] = df['InteractorA'].str.replace('|',',')
df['InteractorB'] = df['InteractorB'].str.replace('|',',')
df = df.assign(InteractorA=df['InteractorA'].str.split(',')).explode('InteractorA')
df = df.assign(InteractorB=df['InteractorB'].str.split(',')).explode('InteractorB')
A = df['InteractorA'].values
B = df['InteractorB'].values
with open('parse_data/parse.biogrid.edgelist', 'w') as f:
    for i in tqdm(range(len(df))):
        if A[i] == str('-'):
            f.writelines("{}\t{}\n".format(B[i],B[i]))
        elif B[i] == str('-'):
            f.writelines("{}\t{}\n".format(A[i],A[i]))
        else:
            f.writelines("{}\t{}\n".format(A[i],B[i]))