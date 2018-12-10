import clmnlab_libs.mvpa_toolkits as mtk
import nilearn.image
import numpy as np
import random
import sys

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.svm import LinearSVC


def perform_analysis():
    results = []

    # in each subjects
    for subj in subj_list:
        # load behavioral information
        labels_list = [
            mtk.get_behavior_data(behav_dir, subj, runs[0], label),
            mtk.get_behavior_data(behav_dir, subj, runs[1], label)
        ]

        # load fMRI file
        img_list = [
            nilearn.image.resample_img(
                nilearn.image.index_img(
                    mtk.load_5d_fmri_image('%s/tvalsLSA.%s.r%02d.nii.gz' % (data_dir, subj, runs[0])),
                    labels['order'] - 1
                ),
                mask.affine, mask.shape, interpolation='nearest'
            )
            for run, labels in zip(runs, labels_list)
        ]

        # make X, y dataset & group info
        data_xs = mtk.masking_fmri_image(nilearn.image.concat_imgs(img_list), mask)
        data_ys = list(labels_list[0]['task_type']) + list(labels_list[1]['task_type'])

        # grouped by run
        group = [1 for _ in labels_list[0]['task_type']] + [2 for _ in labels_list[1]['task_type']]
        cv = GroupKFold(n_splits=2)

        cv_scores = cross_val_score(estimator, data_xs, data_ys, cv=cv, groups=group)

        results.append(cv_scores)

    return results


if __name__ == '__main__':
    random.seed(1210)

    if len(sys.argv) >= 2 and sys.argv[1] in {'move', 'plan', 'color'}:
        label = sys.argv[1]
    else:
        raise ValueError('This code need a label in {move, plan, color}')

    estimator_name = 'svc'
    radius = 8

    # initialize variables
    data_dir = '/clmnlab/IN/MVPA/LSA_tvals/data/'
    mask_dir = '/clmnlab/IN/MVPA/LSA_tvals/masks/'
    behav_dir = '/clmnlab/IN/MVPA/LSS_betas/behaviors/'
    stats_dir = '/clmnlab/IN/MVPA/LSA_tvals/statistics/'

    subj_list = [
        'IN04', 'IN05', 'IN07', 'IN09', 'IN10', 'IN11',
        'IN12', 'IN13', 'IN14', 'IN15', 'IN16', 'IN17',
        'IN18', 'IN23', 'IN24', 'IN25', 'IN26', 'IN28',
        'IN29', 'IN30', 'IN31', 'IN32', 'IN33', 'IN34',
        'IN35', 'IN38', 'IN39', 'IN40', 'IN41', 'IN42',
        'IN43', 'IN45', 'IN46'
    ]

    num_subj = len(subj_list)

    run_number_dict = {
        'move': [3, 5],
        'plan': [3, 4],
        'color': [3, 4],
    }

    if estimator_name is 'svc':
        estimator = LinearSVC()
    elif estimator_name is 'lda':
        estimator = LinearDiscriminantAnalysis(n_components=2)

    runs = run_number_dict[label]

    roi_labels, roi_masks = mtk.load_rois(mask_dir + '*.nii.gz')

    prefix = '%s_%s' % (label, estimator_name)

    with open(stats_dir + '%s_roi_accuracies.tsv' % prefix, 'w') as file:
        file.write(('%s\t'*(num_subj+1) + '%s\n') % ('aal_label', 'mask_size', *subj_list))

    for mask_name, mask in zip(roi_labels, roi_masks):
        scores = perform_analysis()

        with open(stats_dir + '%s_roi_accuracies.tsv' % prefix, 'a') as file:
            file.write(('%s\t'*(num_subj+1) + '%s\n')
                       % (mask_name, np.sum(mask.get_data()), *scores))
