import sys
import time

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from clmnlab_libs.mvpa_toolkits import get_behavior_data, load_rois, load_fmri_image, run_decoding_time_series


def _perform_analysis(subj, estimator, run):
    if estimator == 'gnb':
        estimator = GaussianNB()
    elif estimator == 'svc':
        estimator = Pipeline([
            ('scale', StandardScaler()),
            ('svc', LinearSVC())
        ])

    # load behavioral data
    labels = get_behavior_data(behav_dir, subj, run, label, contain_groups=(1,))

    # load fmri data
    img = load_fmri_image(data_dir, subj, run, labels)
    y = labels['task_type']

    return labels['order'], run_decoding_time_series(estimator, img, y, roi_masks)


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] in {'move', 'color'}:
        label = sys.argv[1]
    else:
        raise ValueError('This code need a label in {move, color}')

    estimator = 'gnb'
    run = 0

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
                else:
                    raise ValueError
            except ValueError:
                raise ValueError('Use these options:\n'
                                 + 'run=run number (1 or 2)'
                                 + 'estimator=estimator name (gnb or svc)')

    if run == 0:
        raise ValueError('This code need a run number (1 or 2). use run=run number.')

    # initialize variables
    data_dir = '/clmnlab/IN/MVPA/LSS_betas/data/'
    behav_dir = '/clmnlab/IN/MVPA/LSS_betas/behaviors/'
    result_dir = '/clmnlab/IN/MVPA/LSS_betas/accuracy_map/'
    stats_dir = '/clmnlab/IN/MVPA/LSS_betas/statistics/'
    roi_dir = '/clmnlab/IN/MVPA/LSS_betas/rois/'

    subj_list = [
        'IN04', 'IN05', 'IN07', 'IN09', 'IN10', 'IN11',
        'IN12', 'IN13', 'IN14', 'IN15', 'IN16', 'IN17',
        'IN18', 'IN23', 'IN24', 'IN25', 'IN26', 'IN28',
        'IN29', 'IN30', 'IN31', 'IN32', 'IN33', 'IN34',
        'IN35', 'IN38', 'IN39', 'IN40', 'IN41', 'IN42',
        'IN43', 'IN45', 'IN46'
    ]

    num_subj = len(subj_list)

    roi_labels, roi_masks = load_rois(file_regex_str=roi_dir + 'searchlight_results/*.nii')

    results = []
    for subj in subj_list:
        results.append(_perform_analysis(subj, estimator, run))
        print(subj, 'finished...')
        time.sleep(5)

    with open(stats_dir + 'decoding_accuracy_%s_%s_run%d.csv' % (label, estimator, run), 'w') as file:
        file.write(('subj,roi_name,' + ('trial_%d,' * 143) + 'trial_%d\n') % (*list(range(1, 145)),))

        for subj, (order, res) in zip(subj_list, results):
            response = [-1] * 144
            for roi_name, roi_corrects in zip(roi_labels, res):
                for idx, correct in zip(order, roi_corrects):
                    response[idx-1] = correct

                file.write(('%s,%s,' + ('%d,' * 143) + '%d\n') % (subj, roi_name, *response))
