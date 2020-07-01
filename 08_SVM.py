# -*- coding: utf-8 -*-
"""
Created on Sat May 26 03:51:33 2020

@author: Marc Jerrone Castro
"""
import pandas as pd
from modules.utils import lload,tokenize_labels, label_unlabbeled_split, MLevaluation
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,make_scorer,f1_score,matthews_corrcoef,precision_score,recall_score, roc_auc_score

#Define Confusion Matrix Scorer
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]

scores = {'Matthews': make_scorer(matthews_corrcoef),'Accuracy':make_scorer(accuracy_score),
      'F1':make_scorer(f1_score),'Precision':make_scorer(precision_score),
      'Recall':make_scorer(recall_score),'AUC':make_scorer(roc_auc_score),
      'tp' : make_scorer(tp), 'tn' : make_scorer(tn),
      'fp' : make_scorer(fp), 'fn' : make_scorer(fn)}

label_map = lload('label_data/biogrid.labels',hashed = False)

unlabelled, labelled_dict, KT_index = tokenize_labels(label_map)
DT_index, NDT_index = KT_index

modules = ['w2v', 'n2v1', 'n2v2']
etype = ['vanilla', 'g2v', 'g2v_concat', 'VanillaG2V']
eclass = ['bio', 'emb', 'bioemb']

for ec in eclass:
    if ec == 'bio':

        data = pd.read_csv('parse_data/parse.biogridv2.bio', delimiter = '\t' ,skiprows= [0] , header = None)
        lb_data, ulb_data = label_unlabbeled_split(data, labelled_dict, unlabelled)

        start_indices = 2
        classifier = SVC(kernel = 'rbf', random_state = 0, probability = True)
        sf = ec

        print('\n\n################',ec,'################')
        acc, Xpred = MLevaluation(lb_data, ulb_data, start_indices, 0.25, classifier,
                                  scores, savePredictions = True, saveformat = sf, lmap = label_map )
    if ec != 'bio':
        start_indices = 1
        for et in etype:
            if et == 'VanillaG2V':
                data = pd.read_csv('emb_data/g2v.{}'.format(ec), delimiter = " ", header = None, skiprows = [0])
                classifier = SVC(kernel = 'rbf', random_state = 0, probability = True)
                sf = (ec,et,'vanilla')
                lb_data, ulb_data = label_unlabbeled_split(data, labelled_dict, unlabelled)
                print('\n\n################',ec,et,'################')
                acc, Xpred = MLevaluation(lb_data, ulb_data, start_indices, 0.25, classifier,
                                          scores, savePredictions = True, saveformat = sf, lmap = label_map )
            else:
                for mod in modules:
                    if et != 'vanilla':
                        data = pd.read_csv('emb_data/{}_to_{}.{}'.format(mod, et, ec), delimiter = " ", header = None, skiprows = [0])
                    else:
                        data = pd.read_csv('emb_data/{}.{}'.format(mod, ec), delimiter = " ", header = None, skiprows = [0])

                    lb_data, ulb_data = label_unlabbeled_split(data, labelled_dict, unlabelled)


                    classifier = SVC(kernel = 'rbf', random_state = 0, probability = True)
                    sf = (ec,et,mod)
                    print('\n\n################',ec,et,mod,'################')
                    acc, Xpred = MLevaluation(lb_data, ulb_data, start_indices, 0.25, classifier,
                                              scores, savePredictions = True, saveformat = sf, lmap = label_map )





