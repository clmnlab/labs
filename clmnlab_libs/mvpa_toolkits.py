import glob
import nilearn.decoding
import nilearn.image
import pandas as pd
import random

from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB


def load_rois(file_regex_str):
    fnames = glob.glob(file_regex_str)
    fnames.sort()

    labels = []
    masks = []

    for fname in fnames:
        masks.append(nilearn.image.load_img(fname))
        label_name = fname.split('/')[-1].replace('nii.gz', '').replace('.nii', '')
        labels.append(label_name)

    return labels, masks


def get_behavior_data(folder_name, subj, run_number, label, stratified_group=False):
    
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
    
    behav_df = pd.read_csv(folder_name + '%s_behav.csv' % subj, index_col=0)
    behav_df = behav_df[behav_df['run'] == run_number]
    
    labels = behav_df.loc[:, ['run', 'order', 'group']]
    labels['task_type'] = behav_df['%s_type' % label]
    
    if stratified_group is True:
        labels['group'] = _stratified_group(labels)
    
    return labels


def load_fmri_image(folder_name, subj, run_number, labels):
    img = nilearn.image.load_img(folder_name + 'betasLSS.%s.r0%d.nii.gz' % (subj, run_number))
    img = nilearn.image.index_img(img, labels['order'] - 1)
    img = nilearn.image.clean_img(img)

    return img


def masking_fmri_image(fmri_imgs, mask_img):
    return nilearn.masking.apply_mask(fmri_imgs, mask_img)


def get_full_mask(data_dir):
    mask_path = data_dir + 'full_mask.group33.nii.gz'
    mask_img = nilearn.image.load_img(mask_path)

    return mask_img


def run_searchlight(full_mask, X, y, group, estimator='svc'):
    if estimator is 'gnb':
        estimator = GaussianNB()
        
    cv = GroupKFold(n_splits=2)
        
    searchlight = nilearn.decoding.SearchLight(
        full_mask,
        radius=8,
        estimator=estimator,
        n_jobs=4,
        verbose=False,
        cv=cv
    )
    
    searchlight.fit(X, y, group)
    scores = searchlight.scores_
    
    return nilearn.image.new_img_like(full_mask, scores)
