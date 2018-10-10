import nilearn.image
import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB


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


def perform_analysis(label, mask, runs, estimator='gnb'):
    results = []

    if estimator is 'gnb':
        estimator = GaussianNB()

    for subj in subj_list:
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

        X = masking_fmri_image(nilearn.image.concat_imgs(img_list), mask)
        y = list(labels_list[0]['task_type']) + list(labels_list[1]['task_type'])
        group = [3 for _ in labels_list[0]['degree']] + [4 for _ in labels_list[1]['degree']]

        scores = cross_val_score(estimator, X, y, cv=2, verbose=1, groups=group)
        results.append(np.mean(scores))

    return results


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] in {'move', 'plan', 'color'}:
        label = sys.argv[1]
    else:
        raise ValueError('This code need a label in {move, plan, color}')

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

    with open(stats_dir + '%s_roi_accuracies.csv' % label, 'w') as file:
        file.write(('%s,'*(num_subj+1) + '%s\n') % ('aal_label', 'mask_size', *subj_list))

    run_number_dict = {
        'move': [3, 5],
        'plan': [3, 4],
        'color': [3, 4],
    }

    for name, mask in zip(roi_labels, roi_masks):
        scores = perform_analysis(label, mask, run_number_dict[label])

        with open(stats_dir + '%s_roi_accuracies.csv' % label, 'a') as file:
            file.write(('%s,'*(num_subj+1) + '%s\n')
                       % (name, np.sum(mask.get_data()), *scores))
