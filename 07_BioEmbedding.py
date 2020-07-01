# -*- coding: utf-8 -*-
"""
Created on Sat May 23 03:51:33 2020

@author: Marc Jerrone Castro
"""
import networkx as nx
import pandas as pd
import numpy as np
import time
import random
from tqdm import tqdm
import pickle
import os
from gensim.models import Word2Vec,KeyedVectors
from modules.utils import bioembedder

edges = 'parse_data/bio.biogrid.edgelist'
P = nx.read_edgelist(edges, encoding ='unicode_escape')

### Loading BioProperties ###
bio = pd.read_csv('parse_data/parse.biogridv2.bio',sep = '\t')

### Creating BioEmbeddings ###
modules = ['w2v','n2v1','n2v2']
etype = ['vanilla','g2v','g2v_concat']

for et in etype:
    if et == 'vanilla':
        for mod in modules: #This loads embeddings we generated from the embedding file
            if mod == 'w2v':
                wv = KeyedVectors.load_word2vec_format('emb_data/w2v.emb')
            elif mod == 'n2v1':
                wv = KeyedVectors.load_word2vec_format('emb_data/n2v1.emb')
            else:
                wv = KeyedVectors.load_word2vec_format('emb_data/n2v2.emb')
            print(et,mod)
            savefile = 'emb_data/{}.bioemb'.format(mod)
            bioembedder(savefile, P, bio, wv, htype = 0)
    elif et == 'g2v':
        for mod in modules:
            print(et,mod)
            wv = KeyedVectors.load_word2vec_format('emb_data/{}_to_g2v.emb'.format(mod))
            savefile = 'emb_data/{}_to_g2v.bioemb'.format(mod)
            bioembedder(savefile, P, bio, wv, htype = 0)
    elif et == 'g2v_concat':
        for mod in modules:
            print(et,mod)
            wv = KeyedVectors.load_word2vec_format('emb_data/{}_to_g2v_concat.emb'.format(mod))
            savefile = 'emb_data/{}_to_g2v_concat.bioemb'.format(mod)
            bioembedder(savefile, P, bio, wv, htype = 1)

#For G2V Implementation on Gensim
wv = KeyedVectors.load_word2vec_format('emb_data/g2v.emb')
savefile = 'emb_data/g2v.bioemb'
bioembedder(savefile, P, bio, wv, htype = 0)


