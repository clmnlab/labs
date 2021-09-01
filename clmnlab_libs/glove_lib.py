import getpass
import os
from os.path import join, dirname, getsize, exists
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
# markers = [next(mks) for i in df["category"].unique()]
# import psutil

import scipy
import statsmodels.stats.multitest
from statsmodels.sandbox.stats.multicomp import multipletests

import sys
import plotly as py
import pickle
import pandas as pd

import nilearn
from nilearn import image, plotting

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from random import random as rand

from datetime import date

from sys import platform

class Common:
    
    def __init__(self):

        ###############
        ## Variables ##
        ###############
        ## 현재 날짜, 예) 2021년 8월 30일 -> 20210830
        self.today = date.today().strftime("%Y%m%d")
    
        ## 정규분포 sigma
        self.sig1 = 0.682689492137
        self.sig2 = 0.954499736104
        self.sig3 = 0.997300203937
        
        ## 디렉토리 설정
        self.dir_lib = join('./glove_lib')
        self.dir_mask = join(self.dir_lib,'masks')

        ## nilearn에서 background로 쓸 MNI-152 image
        self.img_bg = join(self.dir_lib,'mni152_2009bet.nii.gz')

        ## Fan ROI
        self.fan_info = pd.read_csv(join(self.dir_mask,'fan280','fan_cluster_net_20200121.csv'), sep=',', index_col=0)

        ## Models
        ### LDA
        self.lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

    ##############
    ## Behavior ##
    ##############
    
    ##########
    ## fMRI ##
    ##########
        
    def fast_masking(self, img, roi):
        ################
        ## Parameters ##
        ################
        ## img : NIFTI image
        ## roi : NIFTI image
        ############
        ## Return ## : (trials, voxels)-dimensional fdata array
        ############
        img_data = img.get_fdata()
        roi_mask = roi.get_fdata().astype(bool)
        if img_data.shape[:3] != roi_mask.shape:
            raise ValueError('different shape while masking! img=%s and roi=%s' % (img_data.shape, roi_mask.shape))
        ## the shape is (n_trials, n_voxels) which is to cross-validate for runs. =(n_samples, n_features)
        return img_data[roi_mask, :].T

class GA(Common):
    def __init__(self):
        super().__init__()
        
        ## experimental properties
        self.list_subj = ['01', '02', '05', '07', '08', '11', '12', '13', '14', '15'
                          ,'18', '19', '20', '21', '23', '26', '27', '28', '29', '30'
                          ,'31', '32', '33', '34', '35', '36', '37', '38', '42', '44']
        self.list_stage = ['early_practice', 'early_unpractice', 'late_practice', 'late_unpractice']
        
        ## labeling with target position
        # 1 - 5 - 25 - 21 - 1 - 25 - 5 - 21 - 25 - 1 - 21 - 5 - 1 - ...
        ##################
        #  1  2  3  4  5 #
        #  6  7  8  9 10 #
        # 11 12 13 14 15 #
        # 16 17 18 19 20 #
        # 21 22 23 24 25 #
        ##################
        self.target_pos = []
        with open(join(self.dir_lib,'targetID.txt')) as file:
            for line in file:
                self.target_pos.append(int(line.strip()))
        self.target_pos = self.target_pos[1:97]
        # self.target_path = list(range(1,13))*8
        del(file, line)
    
    ##############
    ## Behavior ##
    ##############
    
    def convert_ID(self, ID):
        ##################   ##################
        #  1  2  3  4  5 #   #        2       #
        #  6  7  8  9 10 #   #        1       #
        # 11 12 13 14 15 # = # -2 -1  0  1  2 #
        # 16 17 18 19 20 #   #       -1       #
        # 21 22 23 24 25 #   #       -2       #
        ##################   ##################
        x = np.kron(np.ones(5),np.arange(-2,3)).astype(int)
        y = np.kron(np.arange(2,-3,-1),np.ones(5)).astype(int)
        pos = np.array((x[ID-1],y[ID-1]))
        return pos
    
    def calc_mrew(self, behav_datum):
        ################
        ## Parameters ##
        ################
        ## behav_datum : behavioural datum of each subject (.mat file)
        ############
        ## Return ## : mean scores (range 0.-1.) for each block
        ############

        datum = scipy.io.loadmat(behav_datum)
        nS = int(datum['nSampleTrial'][0][0])
        sec_per_trial = 5  ## time spend(second) in each trial
        ntrial = 12
        nblock = 8
        #ttt = nblock*6 # total number of blocks = 8 blocks/run * 6 runs
        tpr = 97   ## 1 + 12 trials/block * 8 blocks
        nrun = 7

        temp = datum['LearnTrialStartTime'][0]
        idx_editpoint = [i+1 for i,t in enumerate(temp[:-2]) if (temp[i]>temp[i+1])]

        cnt_hit_all = np.zeros((tpr*nrun,nS), dtype=bool)
        for t,ID in enumerate(datum['targetID'][0][idx_editpoint[0]:]):
            pos = datum['boxSize']*self.convert_ID(ID)
            xy = datum['allXY'][:,nS*t:nS*(t+1)] # allXY.shape = (2, 60 Hz * 5 s/trial * 97 trials/run * 7 runs = 203700 frames)
            err = xy - np.ones((2,nS))*pos.T     # err.shape = (2, nS)
            cnt_hit_all[t,:] = (abs(err[0,:]) <= datum['boxSize']*0.5) & (abs(err[1,:]) <= datum['boxSize']*0.5)

        rew_bin = np.zeros((nrun,sec_per_trial*tpr))
        for r in range(nrun):
            temp = cnt_hit_all[tpr*r:tpr*(r+1),:].reshape(nS*tpr,1)
            for i in range(sec_per_trial*tpr):
                rew_bin[r,i] = sum(temp[60*i:60*(i+1)])

        max_score =  nS*ntrial   ## total frames in a block
        temp = rew_bin[:,sec_per_trial:].reshape(nrun*sec_per_trial*ntrial*nblock)
        norm_mrew = np.zeros(nblock*nrun)
        for i in range(nblock*nrun):
            norm_mrew[i] = sum(temp[sec_per_trial*ntrial*i:sec_per_trial*ntrial*(i+1)])/max_score

        return norm_mrew
    
    def make_df_rewards_wide(self, path_behav_dir):
        ################
        ## Parameters ##
        ################
        ## path_behav_dir : Directory location where behavioral files are stored
        ############
        ## Return ## : DataFrame
        ############
        self.df_rewards_wide = pd.DataFrame(columns=['block%02d'%(block+1) for block in range(48)], dtype='float64')
        for subj in self.list_subj:
            for visit in ['early', 'late']:
                suffix = 'fmri' if visit=='early' else('refmri' if visit=='late' else 'invalid')
                subjID = 'GA'+subj if visit=='early' else('GB'+subj if visit=='late' else 'invalid')
                for ii, rew in enumerate(self.calc_mrew(path_behav_dir+'/GA%s-%s.mat'%(subj,suffix))[:48]):
                    self.df_rewards_wide.loc[subjID,'block%02d'%(ii+1)] = rew
        for col in self.df_rewards_wide.columns:
            self.df_rewards_wide[col] = self.df_rewards_wide[col].astype(float)
                    
        return self.df_rewards_wide
    
    def make_df_rewards_long(self, path_behav_dir):
        ################
        ## Parameters ##
        ################
        ## path_behav_dir : Directory location where behavioral files are stored
        ############
        ## Return ## : DataFrame
        ############
        self.df_rewards_long = pd.DataFrame(columns=['subj','visit','block','reward'])
        row = 0
        for subj in self.list_subj:
            for visit in ['early', 'late']:
                suffix = 'fmri' if visit=='early' else('refmri' if visit=='late' else 'invalid')
                rewards = self.calc_mrew(path_behav_dir+'/GA%s-%s.mat'%(subj,suffix))[:48]
                for block, rew in enumerate(rewards):
                    self.df_rewards_long.loc[row,'subj'] = subj
                    self.df_rewards_long.loc[row,'visit'] = visit
                    self.df_rewards_long.loc[row,'block'] = block+1
                    self.df_rewards_long.loc[row,'reward'] = rew
                    row += 1
        self.df_rewards_long.block = self.df_rewards_long.block.astype(int)
        self.df_rewards_long.reward = self.df_rewards_long.reward.astype(float)
        
        return self.df_rewards_long
    
    ##########
    ## fMRI ##
    ##########
    
    def load_fan(self):
        ## load fan_imgs
        self.fan_imgs={}
        path_list = glob(join(self.dir_mask,'fan280','*.nii.gz'))
        for path in path_list:
            temp = path.split('/')[-1].replace('.nii.gz', '')
            fname = temp.split('.')[-1]
            self.fan_imgs[fname] = nilearn.image.load_img(path)

    def load_beta(self, fname):
        ################
        ## Parameters ##
        ################
        ## fname : Full name of a file including a path
        ############
        ## Return ## : 4-D nilearn image
        ############
        
        ## betasLSS.G???.r0?.nii.gz

        ## load betas
        temp = nilearn.image.load_img(join(fname))

        ## We suppose to exclude the first slice from the last dimension of this 4D-image
        beta = nilearn.image.index_img(temp, np.arange(1, 97))

        return beta
