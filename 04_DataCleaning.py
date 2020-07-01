# -*- coding: utf-8 -*-
"""
Created on Sun May  3 20:13:47 2020

@author: user
"""
import networkx as nx
import pandas as pd
import numpy as np
import time
import random
from tqdm import tqdm
import os
from modules.node2vec.utils import *
import modules.node2vec.node2vec as node2vec


###############TTD####################################

#Extracting TTD Drug Target Data
ttd_targets = pd.read_csv("parse_data/parse.TTD.targets", sep = "\t")
ttd_dict = dict(zip(ttd_targets['UniprotID'].values,ttd_targets['TargetType'].values))

#Converting TTD Protein IDs to Uniprot format
ttd_converter = pd.read_csv("input_data/Data Parsing/TTDtoUniprot.tab", sep = "\t")

ttd_parse = pd.DataFrame()
ttd_parse['UID'] = ttd_converter['Entry']
ttd_parse['Inputs'] = ttd_converter['Entry name']
ttd_parse['organism'] = ttd_converter['Organism']

ID = []
TYPE = []
for i in tqdm(range(len(ttd_parse))):
    if ttd_parse['organism'].values[i] == 'Homo sapiens (Human)':
        if ttd_parse['Inputs'].values[i] in ttd_dict.keys():
            if ttd_dict[str(ttd_parse['Inputs'].values[i])] == 'Patented target':
                ID.append(ttd_parse['UID'].values[i])
                TYPE.append('Successful target')
            else:
                ID.append(ttd_parse['UID'].values[i])
                TYPE.append(ttd_dict[str(ttd_parse['Inputs'].values[i])])
ttd_configured = pd.DataFrame()
ttd_dict = dict(zip(ID,TYPE))
ttd_configured['UID'] = ID
ttd_configured['TT']  = TYPE 

##############DRUGBANK##############################

#Extracting DrugBank Target Data
drugbank_all = pd.read_csv('input_data/DrugBank/drugbank.target.data.csv')
drugbank_discontinued = list(pd.read_csv('input_data/DrugBank/drugbank.discontinued.target.csv')['UniProt ID'].values)
drugbank_experimental = list(pd.read_csv('input_data/DrugBank/drugbank.experimental.target.csv')['UniProt ID'].values)
drugbank_research = list(pd.read_csv('input_data/DrugBank/drugbank.research.target.csv')['UniProt ID'].values)
drugbank_succesful = list(pd.read_csv('input_data/DrugBank/drugbank.succesful.target.csv')['UniProt ID'].values)

#Converting DrugBank Label Data to TTD format
db_UID = [] 
db_UID_unknown = []
db_TT = []
db_TT_unknown = []
counts = [0,0,0,0,0]

for x in list(set(drugbank_all['UniProt ID'].values)):
    if x in drugbank_discontinued:
        db_UID.append(x)
        db_TT.append('Discontinued target')
        counts[0] += 1
    elif x in drugbank_succesful:
        db_UID.append(x)
        db_TT.append('Successful target')
        counts[1] += 1
    elif x in drugbank_experimental:
        db_UID.append(x)
        db_TT.append('Clinical Trial target')
        counts[2] += 1
    elif x in drugbank_research:
        db_UID.append(x)
        db_TT.append('Research target')
        counts[3] +=1
    else:
        db_UID_unknown.append(x)
        db_TT_unknown.append('Unlabeled Drug target')
        counts[4]+=1

db_known = dict(zip(db_UID,db_TT))
db_unknown = dict(zip(db_UID_unknown,db_TT_unknown))

'''
With that, we have the following dictionaries:
    db_known => drugbank drug targets with known target classification
    db_unknown => drugbank drug targets with unknown target classification
    ttd_dict => TTD drug targets with reference target classification
'''

##################Data Combination######################

#Isolating redundant entries (NOTE: Drugbank classification was prioritized since it was update more recently)
ttd_unique_data = list(set(ttd_configured['UID'].values) - set(db_UID)) #Isolates TTD unique entries with reference to DrugBank entries
unknown = list(set(db_UID_unknown) - set(ttd_targets)) #Selects unclassified drug targets that are both unknown to TTD and DrugBank

#Initializing Combined Data DataFrame with DrugBank Entries of known Classification
combined_data = pd.DataFrame({'UID':list(db_known.keys()),'TT':list(db_known.values())})

#Add in TTD unique entries
combined_data = pd.concat([pd.DataFrame({'UID':ttd_unique_data,'TT':list(ttd_dict[x] for x in ttd_unique_data)}),combined_data], ignore_index = True)

#As we have nodes which is composed of unknown classification for drug targets, it is within the interest of the researcher to classify them as unlabeled drug targets
combined_data = pd.concat([pd.DataFrame({'UID':unknown,'TT':list(db_unknown[x] for x in unknown)}),combined_data], ignore_index = True)

#################Data Removal############################
'''
Since some proteins are incapable of retrieving nodes from the UniprotDB 
we should parse them from the dataset as this would play a key role in
our analysis later on
'''

#We first create a label file for proteins that are within our PPI dataset
P = nx.read_edgelist('parse_data/parse.biogrid.edgelist', encoding = 'unicode_escape')
nodes = list(P.nodes())
cd_dict = dict(zip(list(combined_data['UID'].values),list(combined_data['TT'].values)))
bg_with_labels = list(set(combined_data['UID'].values) & set(nodes))

#We then combine our known labels to the entire PPI
BG_nodes = [] #22835 datapoints
BG_labels = []
for x in nodes:
    if x in bg_with_labels:
        BG_nodes.append(x)
        BG_labels.append(cd_dict[x])
    else:
        BG_nodes.append(x)
        BG_labels.append('NonDrug target')
all_biogrid_labels = pd.DataFrame({'UID':BG_nodes, 'TT':BG_labels})

with open('parse_data/all.biogrid.labels', 'w') as f:
    for i in tqdm(range(len(all_biogrid_labels))):
        f.writelines("{}\t{}\n".format(all_biogrid_labels['UID'].values[i],all_biogrid_labels['TT'].values[i]))


#Since we will also be using a dataset which utilizes a different data source, thus we must create another label file
bio_data = pd.read_csv('parse_data/parse.biogridv2.bio', sep = '\t')
bio_nodes = list(bio_data['UniprotID']) #22788 datapoints

BG_bionodes = [] #22835 datapoints
BG_biolabels = []
for x in bio_nodes:
    if x in bg_with_labels:
        BG_bionodes.append(x)
        BG_biolabels.append(cd_dict[x])
    else:
        BG_bionodes.append(x)
        BG_biolabels.append('NonDrug target')
bio_biogrid_labels = pd.DataFrame({'UID':BG_bionodes, 'TT':BG_biolabels})

with open('parse_data/bio.biogrid.labels', 'w') as f:
    for i in tqdm(range(len(bio_biogrid_labels))):
        f.writelines("{}\t{}\n".format(bio_biogrid_labels['UID'].values[i],bio_biogrid_labels['TT'].values[i]))
        
#We must also generate a new edgelist which only contains nodes that have biological properties
with open('parse_data/bio.biogrid.edgelist', 'w') as filehandle:
    '''
    This function aims to save a copy of the data that removes all entries without biological properties.
    '''
    edg = set(P.edges())
    for mems in list(edg):
      first = mems[0]
      second = mems[1]
      if (first in bio_nodes) and (second in bio_nodes):
          filehandle.writelines("%s\t%s\n" % (first,second))
