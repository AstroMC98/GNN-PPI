# -*- coding: utf-8 -*-
"""
Created on Sat May 23 12:55:25 2020

@author: Marc Jerrone Castro
"""

import networkx as nx
import pandas as pd
import stellargraph as sg
from modules.utils import tokenize_labels, lload, label_unlabbeled_split, GCN_builder, GSAGE_builder
from sklearn.preprocessing import StandardScaler

edges = 'parse_data/bio.biogrid.edgelist'
P = nx.read_edgelist(edges, encoding ='unicode_escape')

label_map = lload("label_data/biogrid.labels")
unlabelled, labelled_dict, KT_index = tokenize_labels(label_map)
DT_index, NDT_index = KT_index

#mods = ['gsage']
mods = ['gcn','gsage']
files = ['bio.emb', 'w2v_to_g2v.bioemb', 'g2v.emb', 'g2v.bioemb']
ext = ['bio','w2g-bioemb', 'g2v', 'g2v-bioemb']

for mod in mods:
    for f in range(len(files)):
        data = pd.read_csv('emb_data/{}'.format(files[f]),  delimiter = ' ' ,skiprows= [0] , header = None)

        sc = StandardScaler()
        data.iloc[:,1:] = sc.fit_transform(data.iloc[:,1:])
        data.loc[:,'Labels'] = [label_map[x] for x in data[0].values]
        unconnected_proteins = list(set(data[0].values)-set(P.nodes))

        data = data[~data[0].isin(unconnected_proteins)]
        lb_data, ulb_data = label_unlabbeled_split(data, labelled_dict, unlabelled)

        vectors = data.set_index(0)
        graph = sg.StellarGraph.from_networkx(P, node_features = vectors.iloc[:,:-1])
        subjects = vectors['Labels']
        subjects.fillna("Unlabelled target", inplace = True)

        lb_subjects = subjects[subjects != 'Unlabelled target']
        ulb_subjects = subjects[subjects == 'Unlabelled target']

        accuracy,precision,recall,aucs,mats = [],[],[],[],[]

        fmt = ext[f]
        print('{} - {}'.format(mod,fmt))
        if mod == 'gcn':
            scores = GCN_builder(subjects, lb_subjects, ulb_subjects, graph, fmt, generate_embedding = False)
        elif mod =='gsage':
            scores = GSAGE_builder(subjects, lb_subjects, ulb_subjects, graph, fmt, generate_embedding = False)
        accuracy[len(accuracy):], precision[len(precision):], recall[len(recall):], aucs[len(aucs):], mats[len(mats):] = tuple(zip(scores))

        for i in range(2):
            if mod =='gcn':
                scores = GCN_builder(subjects, lb_subjects, ulb_subjects, graph, fmt, generate_embedding = False)
            elif mod =='gsage':
                scores = GSAGE_builder(subjects, lb_subjects, ulb_subjects, graph, fmt, generate_embedding = False)
            accuracy[len(accuracy):], precision[len(precision):], recall[len(recall):], aucs[len(aucs):], mats[len(mats):] = tuple(zip(scores))

        acc = stat.mean(accuracy)
        pcs = stat.mean(precision)
        rec = stat.mean(recall)
        auc = stat.mean(aucs)
        mat = stat.mean(mats)

        acc_std = stat.stdev(accuracy)
        pcs_std = stat.stdev(precision)
        rec_std = stat.stdev(recall)
        auc_std = stat.stdev(aucs)
        mat_std = stat.stdev(mats)

        with open('evaluation_models/{}.{}.results'.format(mod,ext[f]), 'w') as fh:
            print("\nTest Set Metrics:")
            print("Accuracy: {:0.4f} ± {:0.4f}".format(acc,acc_std))
            fh.writelines("Accuracy: {:0.4f} ± {:0.4f}\n".format(acc,acc_std))
            print("Precision: {:0.4f} ± {:0.4f}".format(pcs,pcs_std))
            fh.writelines("Precision: {:0.4f} ± {:0.4f}\n".format(pcs,pcs_std))
            print("Recall: {:0.4f} ± {:0.4f}".format(rec,rec_std))
            fh.writelines("Recall: {:0.4f} ± {:0.4f}\n".format(rec,rec_std))
            print("AUC: {:0.4f} ± {:0.4f}".format(auc,auc_std))
            fh.writelines("AUC: {:0.4f} ± {:0.4f}\n".format(auc,auc_std))
            print("MCC: {:0.4f} ± {:0.4f}".format(mat,mat_std))
            fh.writelines("MCC: {:0.4f} ± {:0.4f}\n".format(mat,mat_std))




