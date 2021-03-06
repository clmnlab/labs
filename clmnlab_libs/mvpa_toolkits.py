import glob
import nilearn.decoding
import nilearn.image
import numpy as np
import pandas as pd
import random

from clmnlab_libs.spatiotemporal_searchlight import SpatioTemporalSearchLight
from sklearn.model_selection import GroupKFold, cross_val_score, LeaveOneOut, ShuffleSplit
from sklearn.naive_bayes import GaussianNB


class BalancedShuffleSplit:
    def __init__(self, n_splits=2):
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        X = np.array(X)
        y = np.array(y)
        
        classes_idx = [np.where(y==cls)[0] for cls in np.unique(y)]
        size = min(map(len, classes_idx)) // 2

        train_idxs = np.array([], dtype=int)
        test_idxs = np.array([], dtype=int)
        
        for idxs in classes_idx:
            np.random.shuffle(idxs)
            
            train_idxs = np.append(train_idxs, idxs[:size])
            test_idxs = np.append(test_idxs, idxs[-size:])
        
        yield train_idxs, test_idxs
        yield test_idxs, train_idxs

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


def load_rois(file_regex_str):
    fnames = glob.glob(file_regex_str)
    fnames.sort()

    labels = []
    masks = []

    for fname in fnames:
        masks.append(nilearn.image.load_img(fname))
        label_name = fname.split('/')[-1].replace('.nii.gz', '').replace('.nii', '')
        labels.append(label_name)

    return labels, masks


def get_behavior_data(folder_name, subj, run_number, label, stratified_group=False, contain_groups=None):
    
    def _stratified_group(labels):
        result = []
        for i, l in labels.groupby('group').count()['order'].iteritems():
            split_idx = int(l / 2)
            if l % 2 == 1:
                if random.random() > 0.5:
                    split_idx += 1

            result.extend([1] * split_idx)
            result.extend([2] * (l-split_idx))
            
        return result

    if type(contain_groups) == int:
        contain_groups = (contain_groups,)
    
    behav_df = pd.read_csv(folder_name + '%s_behav.csv' % subj, index_col=0)
    behav_df = behav_df[behav_df['run'] == run_number]
    
    labels = behav_df.loc[:, ['run', 'order', 'group']]

    if label == 'new_move':
        labels['task_type'] = [1 if de < -15.0 else 3 if de > 15.0 else 2 for de in behav_df['degree_move']]
    else:
        labels['task_type'] = behav_df['%s_type' % label]
    
    if stratified_group is True:
        labels['group'] = _stratified_group(labels)
    if contain_groups is not None:
        labels = labels[labels['group'].isin(contain_groups)]
    
    return labels


def load_fmri_image(folder_name, subj, run_number, labels, standardize=True):
    img = nilearn.image.load_img(folder_name + 'betasLSS.%s.r0%d.nii.gz' % (subj, run_number))
    img = nilearn.image.index_img(img, labels['order'] - 1)

    if standardize:
        img = nilearn.image.clean_img(img)

    return img


def load_5d_fmri_image(fname):
    img = nilearn.image.load_img(fname)
    shape = img.shape

    if len(shape) == 5 and shape[3] == 1:
        reshape_img = nilearn.image.new_img_like(img, img.get_data().reshape([shape[0], shape[1], shape[2], shape[4]]))
        assert np.sum(img.get_data()[:, :, :, 0, -1] - reshape_img.get_data()[:, :, :, -1]) == 0.0
    elif len(shape) == 4:
        print('WARNING: %s is not 5D image! This function will return the original one.' % fname)
        reshape_img = img
    else:
        raise NotImplementedError('%s has an unknown shape %s.' % (fname, shape))

    return reshape_img


def average_N_in_4d_image(img, n=10):
    shape = img.shape

    assert len(shape) == 4
    assert shape[-1] % n == 0

    data = img.get_data()
    data = np.mean(data.reshape((shape[0], shape[1], shape[2], shape[3] // n, n)), axis=4)

    mean_img = nilearn.image.new_img_like(img, data)

    return mean_img


def masking_fmri_image(fmri_imgs, mask_img):
    return nilearn.masking.apply_mask(fmri_imgs, mask_img)


def standardize_trial_wise(img):
    img_data_means = np.mean(np.mean(np.mean(img.get_data(), axis=0), axis=0), axis=0)
    img_data_stds = np.std(np.std(np.std(img.get_data(), axis=0), axis=0), axis=0)
    return nilearn.image.new_img_like(img, (img.get_data()-img_data_means)/img_data_stds)


def standardize_session_wise(img):
    img_data_mean = np.mean(img.get_data())
    img_data_std = np.std(img.get_data())
    return nilearn.image.new_img_like(img, (img.get_data()-img_data_mean)/img_data_std)


def get_full_mask(data_dir, fname='full_mask.group33.nii.gz'):
    mask_path = data_dir + fname
    mask_img = nilearn.image.load_img(mask_path)

    return mask_img


def run_searchlight(full_mask, X, y, group=None, group_k=2, radius=8, estimator='svc', cv=None, chance_level=0):
    if estimator is 'gnb':
        estimator = GaussianNB()
        
    if cv is None:
        cv = GroupKFold(n_splits=group_k)
        
    searchlight = nilearn.decoding.SearchLight(
        full_mask,
        radius=radius,
        estimator=estimator,
        n_jobs=1,
        verbose=False,
        cv=cv
    )
    
    searchlight.fit(X, y, group)
    scores = searchlight.scores_ - chance_level
    
    return nilearn.image.new_img_like(full_mask, scores)


def run_spatiotemporal_searchlight(full_mask, X, y, group, estimator, cv, radius=8, chance_level=0):
    searchlight = SpatioTemporalSearchLight(
        full_mask,
        radius=radius,
        estimator=estimator,
        cv=cv
    )

    searchlight.fit(X, y, group)
    scores = searchlight.scores_ - chance_level

    return nilearn.image.new_img_like(full_mask, scores)


def decoding_with_time_series(estimator, X, y):
    cv = LeaveOneOut()
    return cross_val_score(estimator, X, y, cv=cv)


def run_decoding_time_series(estimator, img, y, roi_masks):
    results = []

    for mask in roi_masks:
        X = masking_fmri_image(img, mask)
        results.append(tuple(decoding_with_time_series(estimator, X, y)))

    return results


def run_roi_based_mvpa(estimator, img, y, roi_masks, cv_method, cv_splits=2, n_iter=1):
    results = []

    if cv_method == 'random':
        cv = ShuffleSplit(n_splits=cv_splits)
    elif cv_method == 'balanced':
        cv = BalancedShuffleSplit(n_splits=cv_splits)
    else:
        raise ValueError('This cv method (%s) is not supported' % cv_method)

    for mask in roi_masks:
        X = masking_fmri_image(img, mask)
        results.append([])

        for _ in range(n_iter):
            results[-1].append(np.mean(cross_val_score(estimator, X, y, cv=cv)))

    return results
