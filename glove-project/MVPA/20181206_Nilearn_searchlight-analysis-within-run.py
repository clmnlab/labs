import nilearn.image
import random
import sys

from collections import Counter
from clmnlab_libs.mvpa_toolkits import run_searchlight, get_full_mask, load_5d_fmri_image, average_N_in_4d_image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


if __name__ == '__main__':
    random.seed(1206)

    estimator_name = 'lda'
    radius = 6
    label = 'path'

    if len(sys.argv) >= 2:
        for argv in sys.argv[1:]:
            try:
                opt, value = argv.split('=')
                if opt == 'label' and value in {'pos', 'path'}:
                    label = value
                else:
                    raise ValueError
            except ValueError:
                raise ValueError('Use these options:\n'
                                 + '  label=label name (pos or path)')

    # initialize variables
    data_dir = '/clmnlab/GA/MVPA/TENT_pb02/data/'
    behav_dir = '/clmnlab/GA/MVPA/TENT_pb02/behaviors/'
    result_dir = '/clmnlab/GA/MVPA/TENT_pb02/accuracy_map/'
    stats_dir = '/clmnlab/GA/MVPA/TENT_pb02/statistics/'

    subj_list = [
        'GA01', 'GA02', 'GA05', 'GA07', 'GA08',
        'GA11', 'GA12', 'GA13', 'GA14', 'GA15',
        'GA18', 'GA19', 'GA20', 'GA21', 'GA23',
        'GB01', 'GB02', 'GB05', 'GB07', 'GB08',
        'GB11', 'GB12', 'GB13', 'GB14', 'GB15',
        'GB18', 'GB19', 'GB20', 'GB21', 'GB23'
    ]

    num_subj = len(subj_list)
    runs = [1, 2, 3, 4, 5, 6, 7]
    group = [(i // 12) + 1 for i in range(96)]

    # estimator initialize
    if estimator_name == 'lda':
        estimator = LinearDiscriminantAnalysis()
    else:
        raise ValueError('!! %s is unknown estimator name' % estimator_name)

    # label initialize
    if label == 'path':
        # all the 12 paths is repeated
        labels = list(range(1, 13)) * 8

        # set chance level
        chance_level = 1 / 12

    elif label == 'pos':
        # read behav file - all the same within subjects
        with open(behav_dir + 'targetID.txt', 'r') as file:
            labels = file.readlines()

        # use only 2 ~ 97 lines - first line is dummy, all the sessions has same order
        labels = [int(l.strip()) for l in labels[1:97]]

        # set chance level
        chance_level = 1 / 4

        # assertion error if all the classes is not the same
        assert set(Counter(labels).values()) == {24}
    else:
        raise ValueError('!! %s is unknown label name' % label)

    mask_img = get_full_mask('/clmnlab/GA/MVPA/fullmask_GAGB/', 'full_mask.GAGB01to19.nii.gz')

    for subj in subj_list:
        print('starting run %s' % subj)

        for run in runs:
            img = load_5d_fmri_image(data_dir + 'tvals.%s.r%02d.nii.gz' % (subj, run))
            img = average_N_in_4d_image(img, n=10)
            img = nilearn.image.index_img(img, range(1, 97))

            X = img
            y = labels

            searchlight_img = run_searchlight(mask_img, X, y, group,
                                              group_k=8, radius=radius, estimator=estimator, chance_level=chance_level)
            searchlight_img.to_filename(
                result_dir + '%s/%s_r%d_within-run%d_%s_%s.nii.gz' % (label, subj, radius, run, estimator_name, label)
            )
