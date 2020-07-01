# -*- coding: utf-8 -*-
"""
Created on Sun May  3 12:13:27 2020

@author: user
"""
import networkx as nx
import pandas as pd
import numpy as np
import time
import random
from tqdm import tqdm
from modules.node2vec.utils import *
import modules.node2vec.node2vec as node2vec
from modules.utils import subgrapher, average_g2v, concat_g2v

edges = 'parse_data/bio.biogrid.edgelist'
P = nx.read_edgelist(edges, encoding ='unicode_escape')

output_W2V_bio = 'emb_data/w2v.emb'
dimensions = 128
walkLength = 20
totalWalks = 10
windowSize = 10
itterations = 10

########################################################
########### Random Walks and Making a Corpus ###########
corpus = []
nodes = list(P.nodes())
print('Walk iterations:')
for walk_iter in range(totalWalks):
    print('\n',str(walk_iter + 1),'/', str(totalWalks))
    random.shuffle(nodes)
    for xx in tqdm(nodes):
        walk = [xx]
        while len(walk) < walkLength:
            cur = walk[-1]
            cur_nbrs = sorted(P.neighbors(cur))
            rand_vertex = random.choice(cur_nbrs)
            walk.append(rand_vertex)
        corpus.append(walk)
        
##################################################################################################
########### Deep Learning using word2vec, node2vec, and my implementation of graph2vec ###########
from gensim.models import Word2Vec,KeyedVectors
from gensim.models.fasttext import FastText

corpus = [list(map(str, walk)) for walk in corpus]

'''
Using the word2vec skip-gram model
'''
print("\n\nStarting Word2Vec implementation")
time1 = time.time()
wv_model = Word2Vec(corpus, size=dimensions, window=windowSize, min_count=0, sg=1, workers=-1, iter=itterations)
wv_model.wv.save_word2vec_format(output_W2V_bio)
embed_train_time = time.time() - time1
print('Word2Vec Embedding Learning Time: %.2f s' % embed_train_time)
print(wv_model.wv.vectors.shape)

'''
For our Node2Vec implementation we import the original python package from 
http://snap.stanford.edu/node2vec/
'''
print('\n\nStarting Node2Vec Implementation...')
print('Reading Graph from {}'.format(edges))

pq_vals = [(0.5,2),(2,0.5)]
time1 = time.time()
print('reading graph')
nx_G = read_graph(edges)
for vals in pq_vals:
    p,q = vals
    if p == 0.5:
        savefile = 'emb_data/n2v1.emb'
    elif p == 2:
        savefile = 'emb_data/n2v2.emb'
    print("Computing for p = {} and q = {}".format(p,q))
    G = node2vec.Graph(nx_G, False, p, q)
    G.preprocess_transition_probs()
    print('Simulating Walks')
    walks = G.simulate_walks(totalWalks, walkLength)
    print('Learning Embeddings')
    learn_embeddings(walks, savefile, dimensions, windowSize, itterations)
    embed_train_time = time.time() -time1
    print('Embedding Learning Time: %.2f s' % embed_train_time)
        
     
'''
My implementation of subgraph2vec
'''
print("\n\nStarting SubGraph2Vec implementation")
corpora = []
time_start = time.time()
for tn in tqdm(list(P.nodes)):
    Psub = subgrapher(P,tn,1)
    corpora.append(Psub)
print(time.time() - time_start)

with open('SubgraphCorpora.graph', 'wb') as file:
    pickle.dump(corpora,file)

modules = ['w2v','n2v1','n2v2']
for mod in modules:
    if mod == 'w2v':
        node_emb_file = output_W2V_bio
    if mod == 'n2v1':
        node_emb_file = 'emb_data/n2v1.emb'
    elif mod == 'n2v2':
        node_emb_file = 'emb_data/n2v2.emb'
        
    savefile = 'emb_data/{}_to_g2v.emb'.format(mod)
    concatfile = 'emb_data/{}_to_g2v_concat.emb'.format(mod)

    average_g2v(node_emb_file, P, corpora, savefile)
    concat_g2v(node_emb_file,savefile,concatfile, P)
        
'''
We also create an implementation of graph2vec using biological data as our embeddings
'''
#Note that due to the presence of properties with nominal values, loading it with gensim is impossible
data = pd.read_csv('parse_data/parse.biogridv2.bio', delimiter = '\t' ,skiprows= [0] , header = None)    
data = data.drop([1], axis = 1)
node_emb_file = 'emb_data/bio.emb'
savefile = 'emb_data/bio_to_g2v.emb'
concatfile = 'emb_data/bio_to_g2v_concat.emb'

with open(node_emb_file, 'w') as f:
    f.writelines('{} {}\n'.format(data.shape[0],data.shape[1]))
    for i in range(len(data)):
        f.writelines('{}'.format(data.loc[i,0]))
        for prop in np.arange(2,46):
            f.writelines(' {}'.format(data.loc[i,prop]))
        f.writelines('\n')

data_prop = data.set_index(0)

df = pd.DataFrame(columns = np.arange(0,44), index = list(P.nodes))
for ci in tqdm(range(len(corpora))):
    corp_df = pd.DataFrame(columns = np.arange(0,44))
    for node in corpora[ci].nodes():
        corp_df.loc[str(node)] = dict(zip(np.arange(0,44),data_prop.loc[str(node)]))
    df.loc[str(list(P.nodes)[ci])] = corp_df.mean(axis = 0)
    
with open(savefile, 'w') as f:
    f.writelines('{} {}\n'.format(df.shape[0],df.shape[1]))
    for i in range(len(df)):
        f.writelines('{}'.format(data.loc[i,0]))
        for prop in np.arange(0,44):
            f.writelines(' {}'.format(df.iloc[i,prop]))
        f.writelines('\n')

average_g2v(node_emb_file, P, corpora, savefile)
#concat_g2v(node_emb_file,savefile,concatfile, P)
'''
Note that this yielded to an underfitted model and was thus disregarded 
'''

'''
Gensim implementation of graph2vec via CBOW using Doc2Vec
'''
import gensim

paragraph = []
for Psub in corpora:
    sentences = []
    edges = list(Psub.edges)
    for edge in edges:
        s,e = edge
        sentences.extend([s,e])
    paragraph.append(sentences)

documents = []
for i,page in enumerate(paragraph):
    document = gensim.models.doc2vec.TaggedDocument(page, [i])
    documents.append(document)
    
model = gensim.models.doc2vec.Doc2Vec(vector_size=128, min_count=2, epochs=10)
model.build_vocab(documents)
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

g2v = []
for p in tqdm(paragraph):
    g2v.append(model.infer_vector(p))

with open('emb_data/g2v.emb', 'w') as f:
    f.writelines('{} 128\n'.format(len(P.nodes)))
    for v in range(len(g2v)):
        f.writelines('{}'.format(list(P.nodes)[v]))
        for vect in g2v[v]:
            f.writelines(' {}'.format(vect))
        f.writelines('\n')
        

   