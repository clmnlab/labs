import nilearn.image
import numpy as np
import pandas as pd


subj_list = [
    'GL03', 'GL04', 'GL05', 'GL06', 'GL07',
    'GL08', 'GL09', 'GL10', 'GL11', 'GL12',
    'GL14', 'GL15', 'GL16', 'GL17', 'GL18',
    'GL19', 'GL20', 'GL21', 'GL22', 'GL24',
    'GL25', 'GL26', 'GL27', 'GL29'
]


def get_behavior_data(feedback_condition, runs=(1, 2, 3, 4),
                      fname='/clmnlab/GL/fmri_data/MVPA/behaviors/targetID_fb_GL.tsv'):
    # target, index, run info initialize (in feedback condition)
    info_df = pd.read_csv(fname, delimiter='\t')
    info_df = info_df[info_df.run.isin(runs) & (info_df.feedback == feedback_condition)]
    targets = np.array(info_df['target'])
    indexes = list(info_df['trial'] - 1)
    group = list(info_df['run'])

    return indexes, targets, group


def spatiotemporal_img_reshape(img, num_trials, num_temporals=2):
    if len(img.shape) != 4:
        raise ValueError('img parameter should be 4D, but its %dD' % len(img.shape))

    if img.shape[-1] != num_trials * num_temporals:
        raise ValueError('The last element in img.shape should be num_trials(%d) * num_temporals(%d), but its %d'
                         % (num_trials, num_temporals, img.shape[-1]))

    data = img.get_data().reshape((img.shape[0], img.shape[1], img.shape[2], num_trials, num_temporals))
    return nilearn.image.new_img_like(img, data)