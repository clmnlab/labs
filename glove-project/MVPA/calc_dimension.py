#!/usr/bin/env python
# coding: utf-8

# # Calculation dimensionality (Tang et al., Nat Neurosci, 2019)
# #### Written by Sungshin Kim, 08.26.2019

## Import libraries
import matplotlib.pyplot as plt
import nilearn.decoding
import nilearn.image
import time
import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from scipy import stats
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

## Set path and variables
root_dir = '/Volumes/clmnlab'
data_dir = root_dir + '/GA/MVPA/LSS_pb02_short_duration_new/data/'
behav_dir = root_dir + '/GA/MVPA/LSS_pb02_short_duration/behaviors/'
mask_dir = '/Users/sskim/Documents/Research/AFNI/GA/masks/'
score_dir = '/Volumes/clmnlab/GA/behavior_data/'

subj_list = [
        'GA01', 'GA02', 'GA05', 'GA07', 'GA08',
        'GA11', 'GA12', 'GA13', 'GA14', 'GA15',
        'GA18', 'GA19', 'GA20', 'GA21', 'GA23',
        'GA26', 'GA27', 'GA28', 'GA29', 'GA30',
        'GA31', 'GA32', 'GA33', 'GA34', 'GA35',
        'GA36', 'GA37', 'GA38', 'GA42', 'GA44'
    ]
num_subj = len(subj_list)
n_stims = 4


def fast_masking(img, roi):
    img_data = img.get_data()
    roi_mask = roi.get_data().astype(bool)
    
    if img_data.shape[:3] != roi_mask.shape:
        raise ValueError('different shape while masking! img=%s and roi=%s' % (img_data.shape, roi_mask.shape))
        
    return img_data[roi_mask, :].T



def make_labels(targets,label_types):
    target_types = np.unique(targets)
    labels = []
    for t in targets:
        for k in range(len(target_types)):       
            if t == list(target_type)[k]:
                if label_types[k]==1:
                    v = 1
                else:
                    v = 2     
        labels.append(v)
    return labels


data = {}
for subj in subj_list:
#    for run in range(1, 2):
#        data[subj, run] = nilearn.image.load_img(data_dir + 'betasLSS.shortdur.%s.r0%d.nii.gz' % (subj, run))
    data[subj] = nilearn.image.load_img(data_dir + 'betasLSS.%s.r0%d.nii.gz' % (subj, run))
    print(subj, end='\r')


for key, value in data.items():
    data[key]=nilearn.image.index_img(value, np.arange(1,97))



with open(behav_dir + 'targetID.txt','r') as file:
    targets = file.readlines()
    targets = [int(l.strip()) for l in targets[1:13]]
temp = list(itertools.product([1,2],repeat=n_stims))
label_types = temp[1:-1]
target_types = np.unique(targets)



labels = make_labels(targets,label_types[0])
print(targets)
print(label_types[0])
print(len(labels))
print(labels)


lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
mask_img = nilearn.image.load_img(mask_dir + 'mask.L_M1.p5.nii')
score = {}
mean_score = {}
for subj in subj_list:
    X = fast_masking(data[subj],mask_img)
    index = 0
    for q in label_types:
        y = make_labels(targets,q)
        index = index + 1
        score[subj, index] = cross_val_score(lda,X,np.tile(y,8),cv=8)
        mean_score[subj, index] = np.mean(score[subj, index])
        print('Processed %s and iter %d : score %f'  %(subj, index, np.mean(score[subj, index])))


np.mean(list(mean_score.values()))
cnt = {}
mmean_score = {}
for subj in subj_list:
    cnt[subj] = len([mean_score[subj,k] for k in range(1,15) if mean_score[subj,k]>0.5])
    mmean_score[subj] = np.mean([mean_score[subj,k] for k in range(1,15)])
#print(cnt.values())
#print(list(cnt.values())) 
#print(list(mmean_score.values()))


def import_behav_scores(fname):
    with open(score_dir + fname) as f:
        data = f.readlines()
        data = [float(l.strip()) for l in data]
    return data


rew_tot_n30 = import_behav_scores('rew_amount_total_n30.1D')
lr_n30 = import_behav_scores('learning_rate_n30.1D')
var_nt0_n30 = import_behav_scores('var_amount_nt0_n30.1D')
rew_amount_except_r01 = import_behav_scores('rew_amount_except_r01_n30.1D')
rew_tot_vs_cnt = stats.linregress(list(cnt.values()), rew_tot_n30)   
lr_vs_cnt = stats.linregress(list(cnt.values()), lr_n30)   
rew_tot_vs_mscore = stats.linregress(list(mmean_score.values()), rew_tot_n30)   
rew_vs_mscore = stats.linregress(list(mmean_score.values()), rew_amount_except_r01)   


print(rew_vs_mscore)




