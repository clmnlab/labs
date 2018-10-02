import nilearn.decoding
import nilearn.image
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB


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
    img = nilearn.image.index_img(img, labels[0]['order'] - 1)

    return img


def run_searchlight(mask, X, y, group, radius=8, estimator='gnb'):
    if estimator is 'gnb':
        estimator = GaussianNB()

    cv = KFold(n_splits=2)

    searchlight = nilearn.decoding.SearchLight(
        mask,
        radius=radius,
        estimator=estimator,
        n_jobs=8,
        verbose=False,
        cv=cv
    )

    searchlight.fit(X, y, group)

    return nilearn.image.new_img_like(mask, searchlight.scores_)


def perform_analysis(label, mask, runs):

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

    X = nilearn.image.concat_imgs(img_list)
    y = list(labels_list[0]['task_type']) + list(labels_list[1]['task_type'])
    group = [3 for _ in labels_list[0]['degree']] + [4 for _ in labels_list[1]['degree']]

    searchlight_img = run_searchlight(mask=mask, X=X, y=y, group=group)
    searchlight_img.to_filename(result_dir + '%s_%s_r8_svc_3class.nii.gz' % (subj, label))


if __name__ == '__main__':
    
    # initialize variables
    data_dir = '/clmnlab/IN/MVPA/LSS_betas/data/'
    behavior_dir = '/clmnlab/IN/MVPA/LSS_betas/behaviors/'
    result_dir = '/clmnlab/IN/MVPA/LSS_betas/accuracy_map/'
    stats_dir = '/clmnlab/IN/MVPA/LSS_betas/statistics/'

    subj_list = [
        'IN04', 'IN05', 'IN07', 'IN09', 'IN10', 'IN11',
        'IN12', 'IN13', 'IN14', 'IN15', 'IN16', 'IN17',
        'IN18', 'IN23', 'IN24', 'IN25', 'IN26', 'IN28',
        'IN29', 'IN30', 'IN31', 'IN32', 'IN33', 'IN34',
        'IN35', 'IN38', 'IN39', 'IN40', 'IN41', 'IN42',
        'IN43', 'IN45', 'IN46'
    ]

    num_subj = len(subj_list)

    # load mask file
    mask_path = data_dir + 'full_mask.group33.nii.gz'
    mask_img = nilearn.image.load_img(mask_path)

    run_number_dict = {
        'move': [3, 5],
        'plan': [3, 4]
    }

    for subj in subj_list:
        perform_analysis('move', mask_img, run_number_dict['move'])
        perform_analysis('plan', mask_img, run_number_dict['plan'])
