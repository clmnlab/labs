import scipy.io
import numpy as np

from collections import Counter


def calc_quadrants(data):
    if data.shape != (2, 203700):
        raise ValueError('Unknown data shape, %s' % data.shape)

    result = [
        1 if x >= 0 and y >= 0 else
        2 if x < 0 and y >= 0 else
        3 if x < 0 and y < 0 else
        4 if x >= 0 and y < 0 else
        -1
        for x, y in data.T
    ]

    assert min(result) == 1

    return result


def count_quadrants_in_each_trial(quadrants):
    result = []
    for i in np.arange(0, 203700, 300):
        count = Counter(quadrants[i:i + 300])
        result.append(count)

    return result


def make_behavior_data(input_fname, output_subj_name):
    data = scipy.io.loadmat(behav_dir + input_fname)

    quadrants = calc_quadrants(data['allXY'])
    count_quadrants = count_quadrants_in_each_trial(quadrants)

    with open(results_dir + '%s_all.tsv' % output_subj_name, 'w') as file:
        file.write('trial\tindex\tquadrants\n')
        for i, q in enumerate(quadrants):
            file.write('%d\t%d\t%d\n' % ((i // 300) + 1, (i % 300) + 1, q))

    with open(results_dir + '%s_probs.tsv' % output_subj_name, 'w') as file:
        file.write('trial\tquadrant_1\tquadrant_2\tquadrant_3\tquadrant_4\n')
        for i, count in enumerate(count_quadrants):
            file.write('%d\t%.3f\t%.3f\t%.3f\t%.3f\n' % (
            i + 1, count[1] / 300, count[2] / 300, count[3] / 300, count[4] / 300))

    with open(results_dir + '%s_compact.tsv' % output_subj_name, 'w') as file:
        file.write('trial\tquadrants\n')
        for i, count in enumerate(count_quadrants):
            file.write('%d\t%d\n' % (i + 1, count.most_common(1)[0][0]))


if __name__ == '__main__':
    subj_list = [
        'GA01', 'GA02', 'GA05', 'GA07', 'GA08',
        'GA11', 'GA12', 'GA13', 'GA14', 'GA15',
        'GA18', 'GA19', 'GA20', 'GA21', 'GA23',
        'GA26', 'GA27', 'GA28'
    ]

    behav_dir = '/Volumes/clmnlab/GA/behavior_data/'
    results_dir = '/Volumes/clmnlab/GA/MVPA/TENT_pb02/behaviors/'

    for subj in subj_list:
        make_behavior_data('%s/%s-fmri.mat' % (subj, subj), subj)
        make_behavior_data('%s/%s-refmri.mat' % (subj, subj), subj.replace('GA', 'GB'))
