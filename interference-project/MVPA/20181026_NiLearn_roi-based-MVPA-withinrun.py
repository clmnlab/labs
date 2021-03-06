import sys
import time

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from clmnlab_libs.mvpa_toolkits import get_behavior_data, load_rois, load_fmri_image, run_roi_based_mvpa


def _perform_analysis(subj, estimator, run, label, niter, standardize=True):
    if estimator == 'gnb':
        estimator = GaussianNB()
    elif estimator == 'svc':
        estimator = Pipeline([
            ('scale', StandardScaler()),
            ('svc', LinearSVC())
        ])

    # load behavioral data
    labels = get_behavior_data(behav_dir, subj, run, label)

    # load fmri data
    img = load_fmri_image(data_dir, subj, run, labels, standardize=standardize)
    y = labels['task_type']

    return run_roi_based_mvpa(estimator, img, y, roi_masks, 'balanced', n_iter=niter)


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] in {'move', 'plan', 'color'}:
        label = sys.argv[1]
    else:
        raise ValueError('This code need a label in {move, plan, color}')

    estimator = 'gnb'
    run = 0
    niter = 1
    standardize = True

    if len(sys.argv) >= 3:
        for argv in sys.argv[2:]:
            try:
                opt, value = argv.split('=')
                if opt == 'run':
                    run = int(value)
                elif opt == 'estimator':
                    if value == 'svc':
                        estimator = 'svc'
                    elif value == 'gnb':
                        estimator = 'gnb'
                    else:
                        raise ValueError
                elif opt == 'iter':
                    niter = int(value)
                elif opt == 'standardize':
                    standardize = bool(int(value))
                else:
                    raise ValueError
            except ValueError:
                raise ValueError('Use these options:\n'
                                 + 'run=run number (3, 4 or 5)'
                                 + 'estimator=estimator name (gnb or svc)'
                                 + 'iteration=iteration number (integer)'
                                 + 'standardize=0 or 1 (0=False, 1=True)')

    if run == 0:
        raise ValueError('This code need a run number (1 or 2). use run=run number.')

    # initialize variables
    data_dir = '/clmnlab/IN/MVPA/LSS_betas/data/'
    behav_dir = '/clmnlab/IN/MVPA/LSS_betas/behaviors/'
    result_dir = '/clmnlab/IN/MVPA/LSS_betas/accuracy_map/'
    stats_dir = '/clmnlab/IN/MVPA/LSS_betas/statistics/'
    roi_dir = '/clmnlab/IN/MVPA/LSS_betas/rois/'
    output_filename = 'roi_accuracies_%s_%s_run%d_iter%d_standardize%s.tsv' % (label, estimator, run, niter, standardize)

    subj_list = [
        'IN04', 'IN05', 'IN07', 'IN09', 'IN10', 'IN11',
        'IN12', 'IN13', 'IN14', 'IN15', 'IN16', 'IN17',
        'IN18', 'IN23', 'IN24', 'IN25', 'IN26', 'IN28',
        'IN29', 'IN30', 'IN31', 'IN32', 'IN33', 'IN34',
        'IN35', 'IN38', 'IN39', 'IN40', 'IN41', 'IN42',
        'IN43', 'IN45', 'IN46'
    ]

    num_subj = len(subj_list)

    roi_labels, roi_masks = load_rois(file_regex_str=roi_dir + 'run12_glm/*.nii')

    with open(stats_dir + output_filename, 'w') as file:
        file.write(('subj\t' + ('%s\t' * (len(roi_labels)-1)) + '%s\n') % (*roi_labels,))

    results = []
    for subj in subj_list:
        results.append(_perform_analysis(subj, estimator, run, label, niter))
        print(subj, 'finished...')

        with open(stats_dir + output_filename, 'a') as file:
            file.write(('%s\t' + ('%s\t' * (len(roi_labels)-1)) + '%s\n') % (subj, *results[-1]))

