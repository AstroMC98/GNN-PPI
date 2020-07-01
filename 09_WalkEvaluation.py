# -*- coding: utf-8 -*-
"""
Created on Thu May 21 22:21:12 2020

@author: Marc Jerrone Castro
"""

import pandas as pd
import statistics as stat
import pickle

with open("evaluation_models/bioemb.VanillaG2V.vanilla.results", 'rb') as f:
    eval_file = pickle.load(f)

modules = ['w2v', 'n2v1', 'n2v2']
etype = ['vanilla', 'g2v', 'g2v_concat', 'VanillaG2V']
eclass = ['bio', 'emb', 'bioemb']

df = pd.DataFrame(columns = ['dtype', 'acc', 'acc_std', 'pres', 'pres_std', 'rec', 'rec_std', 'auc', 'auc_std', 'f1', 'f1_std', 'matthews', 'matthews_std'])

for ec in eclass:
    if ec == 'bio':
        fload = open("evaluation_models/b.i.o.results", 'rb')
        file = pickle.load(fload)
        df = df.append({
            'dtype': ec,
            'acc':float('{:0.4f}'.format(stat.mean(file['test_Accuracy'].values))),
            'acc_std':float('{:0.4f}'.format(stat.stdev(file['test_Accuracy'].values))),
            'pres':float('{:0.4f}'.format(stat.mean(file['test_Precision'].values))),
            'pres_std':float('{:0.4f}'.format(stat.stdev(file['test_Precision'].values))),
            'rec':float('{:0.4f}'.format(stat.mean(file['test_Recall'].values))),
            'rec_std':float('{:0.4f}'.format(stat.stdev(file['test_Recall'].values))),
            'auc':float('{:0.4f}'.format(stat.mean(file['test_AUC'].values))),
            'auc_std':float('{:0.4f}'.format(stat.stdev(file['test_AUC'].values))),
            'f1':float('{:0.4f}'.format(stat.mean(file['test_F1'].values))),
            'f1_std':float('{:0.4f}'.format(stat.stdev(file['test_F1'].values))),
            'matthews':float('{:0.4f}'.format(stat.mean(file['test_Matthews'].values))),
            'matthews_std':float('{:0.4f}'.format(stat.stdev(file['test_Matthews'].values))),
            }, ignore_index = True)

    else:
        for et in etype:
            if et == 'VanillaG2V':
                fload = open("evaluation_models/{}.VanillaG2V.vanilla.results".format(ec),'rb')
                file = pickle.load(fload)
                df = df.append({
                    'dtype': '{}-Vanilla-g2v'.format(ec),
                    'acc':float('{:0.4f}'.format(stat.mean(file['test_Accuracy'].values))),
                    'acc_std':float('{:0.4f}'.format(stat.stdev(file['test_Accuracy'].values))),
                    'pres':float('{:0.4f}'.format(stat.mean(file['test_Precision'].values))),
                    'pres_std':float('{:0.4f}'.format(stat.stdev(file['test_Precision'].values))),
                    'rec':float('{:0.4f}'.format(stat.mean(file['test_Recall'].values))),
                    'rec_std':float('{:0.4f}'.format(stat.stdev(file['test_Recall'].values))),
                    'auc':float('{:0.4f}'.format(stat.mean(file['test_AUC'].values))),
                    'auc_std':float('{:0.4f}'.format(stat.stdev(file['test_AUC'].values))),
                    'f1':float('{:0.4f}'.format(stat.mean(file['test_F1'].values))),
                    'f1_std':float('{:0.4f}'.format(stat.stdev(file['test_F1'].values))),
                    'matthews':float('{:0.4f}'.format(stat.mean(file['test_Matthews'].values))),
                    'matthews_std':float('{:0.4f}'.format(stat.stdev(file['test_Matthews'].values))),
                    }, ignore_index = True)

            else:
                for mod in modules:
                    fload = open("evaluation_models/{}.{}.{}.results".format(ec,et,mod),'rb')
                    file = pickle.load(fload)

                    df = df.append({
                        'dtype': '{}-{}-{}'.format(ec,et,mod),
                        'acc':float('{:0.4f}'.format(stat.mean(file['test_Accuracy'].values))),
                        'acc_std':float('{:0.4f}'.format(stat.stdev(file['test_Accuracy'].values))),
                        'pres':float('{:0.4f}'.format(stat.mean(file['test_Precision'].values))),
                        'pres_std':float('{:0.4f}'.format(stat.stdev(file['test_Precision'].values))),
                        'rec':float('{:0.4f}'.format(stat.mean(file['test_Recall'].values))),
                        'rec_std':float('{:0.4f}'.format(stat.stdev(file['test_Recall'].values))),
                        'auc':float('{:0.4f}'.format(stat.mean(file['test_AUC'].values))),
                        'auc_std':float('{:0.4f}'.format(stat.stdev(file['test_AUC'].values))),
                        'f1':float('{:0.4f}'.format(stat.mean(file['test_F1'].values))),
                        'f1_std':float('{:0.4f}'.format(stat.stdev(file['test_F1'].values))),
                        'matthews':float('{:0.4f}'.format(stat.mean(file['test_Matthews'].values))),
                        'matthews_std':float('{:0.4f}'.format(stat.stdev(file['test_Matthews'].values))),
                        }, ignore_index = True)

df.to_csv('plot_files/walk.results')

