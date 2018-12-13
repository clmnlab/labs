import clmnlab_libs.mvpa_toolkits as mtk
import nilearn.image
import numpy as np
import sys

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def initial_images(standardize_trial=False):
    result = {}

    for subj in subj_list:
        labels = mtk.get_behavior_data(behav_dir, subj, run, 'color')

        # load and resampling image
        img = mtk.load_5d_fmri_image(
            data_dir + 'tvalsLSA.%s.r0%d.nii.gz' % (subj, run))
        img = nilearn.image.index_img(img, labels['order'] - 1)

        if standardize_trial:
            temp = mtk.masking_fmri_image(img, mask_img)
            temp = temp - np.tile(np.mean(temp, axis=1), (temp.shape[-1], 1)).T
            img = nilearn.masking.unmask(temp, mask_img)

        img = nilearn.image.resample_img(img, roi_masks[0].affine, roi_masks[0].shape, interpolation='nearest')

        result[subj] = img

    return result


def perform_analysis():
    results = []

    for subj in subj_list:
        labels = mtk.get_behavior_data(behav_dir, subj, run, label)
        imgs = mtk.masking_fmri_image(img_data[subj], mask)

        data_xs = imgs
        data_ys = list(labels['task_type'])
        cv = mtk.BalancedShuffleSplit(n_splits=2)

        cv_scores = cross_val_score(estimator, data_xs, data_ys, cv=cv)

        results.append(list(cv_scores))

    return results


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] in {'move', 'plan', 'color'}:
        label = sys.argv[1]
    else:
        raise ValueError('This code need a label in {move, plan, color}')

    run = 0
    estimator_name = 'svc'

    if len(sys.argv) >= 3:
        for argv in sys.argv[2:]:
            try:
                opt, value = argv.split('=')
                if opt == 'run':
                    run = int(value)
                else:
                    raise ValueError
            except ValueError:
                raise ValueError('Use these options:\n'
                                 + 'run=run number (3, 4 or 5)')

    if run == 0:
        raise ValueError('This code need a run number (1 or 2). use run=run number.')

    # initialize variables
    data_dir = '/clmnlab/IN/MVPA/LSA_tvals/data/'
    behav_dir = '/clmnlab/IN/MVPA/LSS_betas/behaviors/'
    stats_dir = '/clmnlab/IN/MVPA/LSA_tvals/statistics/'
    roi_dir = '/clmnlab/IN/MVPA/LSA_tvals/masks/'
    mask_dir = '/clmnlab/IN/MVPA/LSS_betas/data/'
    output_filename = '%s_%s_roi_accuracies_run%02d.tsv' % (label, estimator_name, run)

    subj_list = [
        'IN04', 'IN05', 'IN07', 'IN09', 'IN10', 'IN11',
        'IN12', 'IN13', 'IN14', 'IN15', 'IN16', 'IN17',
        'IN18', 'IN23', 'IN24', 'IN25', 'IN26', 'IN28',
        'IN29', 'IN30', 'IN31', 'IN32', 'IN33', 'IN34',
        'IN35', 'IN38', 'IN39', 'IN40', 'IN41', 'IN42',
        'IN43', 'IN45', 'IN46'
    ]
    num_subj = len(subj_list)
    mask_img = mtk.get_full_mask(mask_dir)

    if estimator_name == 'svc':
        estimator = LinearSVC()
    elif estimator_name == 'scaled-svc':
        estimator = Pipeline([('scale', StandardScaler()), ('svc', LinearSVC())])

    roi_labels, roi_masks = mtk.load_rois(file_regex_str=roi_dir + 'AAL_ROI_*.nii.gz')
    img_data = initial_images()

    with open(stats_dir + output_filename, 'w') as file:
        file.write(('%s\t'*(num_subj+1) + '%s\n') % ('aal_label', 'mask_size', *subj_list))

    for mask_name, mask in zip(roi_labels, roi_masks):
        scores = perform_analysis()

        with open(stats_dir + output_filename, 'a') as file:
            file.write(('%s\t'*(num_subj+1) + '%s\n')
                       % (mask_name, np.sum(mask.get_data()), *scores))