# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:38:12 2020

@author: user
"""
import networkx as nx
import pandas as pd
import numpy as np
import time
import random
from tqdm import tqdm
import os
import pickle

with open("parse_data/bio.biogrid.family", 'rb') as f:
    family_dict = pickle.load(f)

allfamilies = []
for key,value in family_dict.items():
    for fam in value:
        allfamilies.append(fam)
allfamilies = list(set(allfamilies))

alllabels = pd.read_csv('parse_data/bio.biogrid.labels', sep ='\t', header = None)

drugFamilies = []
for x in list(alllabels[alllabels[1] != 'NonDrug target'][0]):
    fams = family_dict[str(x)]
    for fam in fams:
        drugFamilies.append(fam)
drugFamilies = list(set(drugFamilies))

UT = 0
DT = 0
NDT = 0
label_dict = {}
label_dict2 = {}
for i in range(len(alllabels)):
    if alllabels[1][i] == 'NonDrug target':
        protein = alllabels[0][i]
        protein_fams = family_dict[str(protein)]
        if len(set(drugFamilies).intersection(protein_fams)) == 0:
            label_dict[str(alllabels[0][i])] = 'NonDrug target'
            label_dict2[str(alllabels[0][i])] = 'NonDrug target'
            NDT+=1
        elif len(set(drugFamilies).intersection(protein_fams)) > 0:
            label_dict[str(alllabels[0][i])] = 'Unlabelled target'
            label_dict2[str(alllabels[0][i])] = 'Unlabelled target'
            UT +=1
    else:
        label_dict[str(alllabels[0][i])] = 'Drug target'
        label_dict2[str(alllabels[0][i])] = str(alllabels[1][i])
        DT+=1
print('Drug Targets = {}, NonDrug Targets = {}, Unlabelled = {}'.format(DT,NDT,UT))

with open('label_data/biogrid.labels', 'wb') as outfile:
    pickle.dump(label_dict,outfile, pickle.DEFAULT_PROTOCOL)

with open('label_data/biogrid.specific.labels' , 'wb') as outfile:
    pickle.dump(label_dict,outfile, pickle.DEFAULT_PROTOCOL)