import nilearn.image
import numpy as np
import pandas as pd
import random
import sys

from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC


def load_aal_rois(folder_name):
    roi_masks = []

    for i in range(116):
        roi_mask_img = nilearn.image.load_img(folder_name + 'AAL_ROI_%03d.nii' % (i+1))
        roi_masks.append(roi_mask_img)

    return roi_masks


def get_behavior_data(folder_name, subj, run_number, label_name):

    def _get_labels(fname):
        labels = pd.read_csv(folder_name + '%s.csv' % fname, names=['run', 'degree', 'order'])
        labels = labels.set_index('order').join(
            pd.read_csv(folder_name + '%s_index.csv' % fname, names=['task_type', 'order']).set_index('order'))

        return labels.reset_index(drop=False)

    filename = '%s_run%d_%s' % (subj, run_number, label_name)

    return _get_labels(filename)


def load_fmri_image(folder_name, subj, run_number, labels):
    img = nilearn.image.load_img(folder_name + 'betasLSS.%s.r0%d.nii.gz' % (subj, run_number))
    img = nilearn.image.index_img(img, labels['order'] - 1)
    img = nilearn.image.clean_img(img)

    return img


def masking_fmri_image(fmri_imgs, mask_img):
    return nilearn.masking.apply_mask(fmri_imgs, mask_img)


def averaging_random_3_samples(x_grouped_samples, y_grouped_samples):
    # data reshaping
    datas = [dict() for _ in range(len(x_grouped_samples))]

    for i, (x_samples, y_samples) in enumerate(zip(x_grouped_samples, y_grouped_samples)):
        for x, y in zip(x_samples, y_samples):
            if y not in datas[i]:
                datas[i][y] = []
            datas[i][y].append(x)

    # data averaging
    averaged_x_samples = []
    averaged_y_samples = []
    group = []

    for i, data in enumerate(datas):
        for y, xs in data.items():
            idxs = list(range(len(xs)))
            random.shuffle(idxs)

            averaged_x_samples += [np.mean([xs[i], xs[j], xs[k]], axis=0)
                                   for i, j, k in zip(idxs[0:-2:3], idxs[1:-1:3], idxs[2::3])]
            averaged_y_samples += [y for _ in idxs[2::3]]
            group += [i for _ in idxs[2::3]]

    return averaged_x_samples, averaged_y_samples, group


def cross_validation_with_mix(estimator, X, y, mix=False, group=None):
    if mix is False:
        results = cross_val_score(estimator, X, y, cv=2, groups=group)
    elif mix == 'loocv':
        cv = LeaveOneOut()
        results = cross_val_score(estimator, X, y, cv=cv)
    else:
        results = cross_val_score(estimator, X, y, cv=mix)

    return results


def _perform_analysis(subj, label, mask, runs, estimator, average_iter, mix):
    # load behavioral data
    labels_list = [
        get_behavior_data(behavior_dir, subj, runs[0], label),
        get_behavior_data(behavior_dir, subj, runs[1], label)
    ]

    # load fmri file
    img_list = [
        load_fmri_image(data_dir, subj, runs[0], labels_list[0]),
        load_fmri_image(data_dir, subj, runs[1], labels_list[1]),
    ]

    if average_iter:
        for i in range(average_iter):
            Xs = [masking_fmri_image(img, mask) for img in img_list]
            ys = [list(labels['task_type']) for labels in labels_list]

            X, y, group = averaging_random_3_samples(Xs, ys)
            cv_scores = cross_validation_with_mix(estimator, X, y, mix, group)
            return np.mean(cv_scores)
    else:
        X = masking_fmri_image(nilearn.image.concat_imgs(img_list), mask)
        y = list(labels_list[0]['task_type']) + list(labels_list[1]['task_type'])
        group = [1 for _ in labels_list[0]['degree']] + [2 for _ in labels_list[1]['degree']]

        cv_scores = cross_validation_with_mix(estimator, X, y, mix, group)
        return np.mean(cv_scores)


def perform_analysis(label, mask, runs, estimator='gnb', average_iter=False, mix=False):
    if estimator is 'gnb':
        estimator = GaussianNB()
    if estimator is 'svc':
        estimator = LinearSVC()

    results = Parallel(n_jobs=4)(delayed(_perform_analysis)(subj, label, mask, runs, estimator, average_iter, mix)
                                 for subj in subj_list)

    return results


if __name__ == '__main__':
    random.seed(1008)

    if len(sys.argv) >= 2 and sys.argv[1] in {'move', 'plan', 'color'}:
        label = sys.argv[1]
    else:
        raise ValueError('This code need a label in {move, plan, color}')

    average = False
    mix = False
    estimator = 'gnb'

    if len(sys.argv) >= 3:
        for argv in sys.argv[2:]:
            try:
                opt, value = argv.split('=')
                if opt == 'avg':
                    average = int(value)
                elif opt == 'mix':
                    if value == 'loocv':
                        mix = 'loocv'
                    else:
                        mix = int(value)
                elif opt == 'estimator':
                    if value == 'svc':
                        estimator = 'svc'
                    else:
                        raise ValueError
                else:
                    raise ValueError
            except ValueError:
                raise ValueError('If you want to use options, '
                                 + 'write avg=average_iteration_count OR mix=cross_validation_count OR estimator=svc.\n'
                                 + 'ex) python filename.py avg=100 mix=10 estimator=svc\n'
                                 + 'You can also use mix=loocv, that means Leave-One-Out-Cross-Validation.')

    data_dir = '/clmnlab/IN/MVPA/LSS_betas/data/'
    behavior_dir = '/clmnlab/IN/MVPA/LSS_betas/behaviors/'
    result_dir = '/clmnlab/IN/MVPA/LSS_betas/accuracy_map/'
    stats_dir = '/clmnlab/IN/MVPA/LSS_betas/statistics/'
    roi_dir = '/clmnlab/IN/AFNI_data/masks/AAL_ROI/'

    subj_list = [
        'IN04', 'IN05', 'IN07', 'IN09', 'IN10', 'IN11',
        'IN12', 'IN13', 'IN14', 'IN15', 'IN16', 'IN17',
        'IN18', 'IN23', 'IN24', 'IN25', 'IN26', 'IN28',
        'IN29', 'IN30', 'IN31', 'IN32', 'IN33', 'IN34',
        'IN35', 'IN38', 'IN39', 'IN40', 'IN41', 'IN42',
        'IN43', 'IN45', 'IN46'
    ]
    num_subj = len(subj_list)

    roi_labels = [
        'Precentral_L',
        'Precentral_R',
        'Frontal_Sup_L',
        'Frontal_Sup_R',
        'Frontal_Sup_Orb_L',
        'Frontal_Sup_Orb_R',
        'Frontal_Mid_L',
        'Frontal_Mid_R',
        'Frontal_Mid_Orb_L',
        'Frontal_Mid_Orb_R',
        'Frontal_Inf_Oper_L',
        'Frontal_Inf_Oper_R',
        'Frontal_Inf_Tri_L',
        'Frontal_Inf_Tri_R',
        'Frontal_Inf_Orb_L',
        'Frontal_Inf_Orb_R',
        'Rolandic_Oper_L',
        'Rolandic_Oper_R',
        'Supp_Motor_Area_L',
        'Supp_Motor_Area_R',
        'Olfactory_L',
        'Olfactory_R',
        'Frontal_Sup_Medial_L',
        'Frontal_Sup_Medial_R',
        'Frontal_Med_Orb_L',
        'Frontal_Med_Orb_R',
        'Rectus_L',
        'Rectus_R',
        'Insula_L',
        'Insula_R',
        'Cingulum_Ant_L',
        'Cingulum_Ant_R',
        'Cingulum_Mid_L',
        'Cingulum_Mid_R',
        'Cingulum_Post_L',
        'Cingulum_Post_R',
        'Hippocampus_L',
        'Hippocampus_R',
        'ParaHippocampal_L',
        'ParaHippocampal_R',
        'Amygdala_L',
        'Amygdala_R',
        'Calcarine_L',
        'Calcarine_R',
        'Cuneus_L',
        'Cuneus_R',
        'Lingual_L',
        'Lingual_R',
        'Occipital_Sup_L',
        'Occipital_Sup_R',
        'Occipital_Mid_L',
        'Occipital_Mid_R',
        'Occipital_Inf_L',
        'Occipital_Inf_R',
        'Fusiform_L',
        'Fusiform_R',
        'Postcentral_L',
        'Postcentral_R',
        'Parietal_Sup_L',
        'Parietal_Sup_R',
        'Parietal_Inf_L',
        'Parietal_Inf_R',
        'SupraMarginal_L',
        'SupraMarginal_R',
        'Angular_L',
        'Angular_R',
        'Precuneus_L',
        'Precuneus_R',
        'Paracentral_Lobule_L',
        'Paracentral_Lobule_R',
        'Caudate_L',
        'Caudate_R',
        'Putamen_L',
        'Putamen_R',
        'Pallidum_L',
        'Pallidum_R',
        'Thalamus_L',
        'Thalamus_R',
        'Heschl_L',
        'Heschl_R',
        'Temporal_Sup_L',
        'Temporal_Sup_R',
        'Temporal_Pole_Sup_L',
        'Temporal_Pole_Sup_R',
        'Temporal_Mid_L',
        'Temporal_Mid_R',
        'Temporal_Pole_Mid_L',
        'Temporal_Pole_Mid_R',
        'Temporal_Inf_L',
        'Temporal_Inf_R',
        'Cerebelum_Crus1_L',
        'Cerebelum_Crus1_R',
        'Cerebelum_Crus2_L',
        'Cerebelum_Crus2_R',
        'Cerebelum_3_L',
        'Cerebelum_3_R',
        'Cerebelum_4_5_L',
        'Cerebelum_4_5_R',
        'Cerebelum_6_L',
        'Cerebelum_6_R',
        'Cerebelum_7b_L',
        'Cerebelum_7b_R',
        'Cerebelum_8_L',
        'Cerebelum_8_R',
        'Cerebelum_9_L',
        'Cerebelum_9_R',
        'Cerebelum_10_L',
        'Cerebelum_10_R',
        'Vermis_1_2',
        'Vermis_3',
        'Vermis_4_5',
        'Vermis_6',
        'Vermis_7',
        'Vermis_8',
        'Vermis_9',
        'Vermis_10'
    ]
    roi_masks = load_aal_rois(roi_dir)

    prefix = label
    if average:
        prefix = 'avg%d_%s' % (average, prefix)
    if mix:
        prefix = 'cv%s_%s' % (mix, prefix)
    if estimator != 'gnb':
        prefix = '%s_%s' % (prefix, estimator)

    with open(stats_dir + '%s_roi_accuracies.csv' % prefix, 'w') as file:
        file.write(('%s,'*(num_subj+1) + '%s\n') % ('aal_label', 'mask_size', *subj_list))

    run_number_dict = {
        'move': [3, 5],
        'plan': [3, 4],
        'color': [3, 4],
    }

    for name, mask in zip(roi_labels, roi_masks):
        scores = perform_analysis(label, mask, run_number_dict[label],
                                  average_iter=average, mix=mix, estimator=estimator)

        with open(stats_dir + '%s_roi_accuracies.csv' % prefix, 'a') as file:
            file.write(('%s,'*(num_subj+1) + '%s\n')
                       % (name, np.sum(mask.get_data()), *scores))
