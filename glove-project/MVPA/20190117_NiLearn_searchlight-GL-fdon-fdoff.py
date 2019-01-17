import clmnlab_libs.mvpa_toolkits as mtk
import nilearn.image
import numpy as np
import pandas as pd
import sys

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


if __name__ == '__main__':
    estimator_name = 'lda'
    radius = 6
    label = 'target'
    feedback_on = None

    # get argument variables - feedback on / off
    try:
        if len(sys.argv) >= 2:
            for argv in sys.argv[1:]:
                    opt, value = argv.split('=')
                    if opt == 'feedback' and value in {'1', '0'}:
                        feedback_on = int(value)
                    else:
                        raise ValueError
        else:
            raise ValueError
    except ValueError:
        raise ValueError('Use these options:\n'
                         + '  feedback=(1 or 0)')

    # initialize variables
    data_dir = '/clmnlab/GL/fmri_data/'
    behav_dir = '/clmnlab/GL/fmri_data/MVPA/behaviors/'
    result_dir = '/clmnlab/GL/fmri_data/MVPA/searchlight/'

    subj_list = [
        'GL03', 'GL04', 'GL05', 'GL06', 'GL07',
        'GL08', 'GL09', 'GL10', 'GL11', 'GL12',
        'GL14', 'GL15', 'GL16', 'GL17', 'GL18',
        'GL19', 'GL20', 'GL21', 'GL22', 'GL24',
        'GL25', 'GL26', 'GL27', 'GL29'
    ]

    runs = [1, 2, 3, 4]

    # estimator initialize
    estimator = LinearDiscriminantAnalysis()

    # target, index, run info initialize (in feedback condition)
    info_df = pd.read_csv('/clmnlab/GL/fmri_data/MVPA/behaviors/targetID_fb_GL.tsv', delimiter='\t')
    chance_level = 1 / 4

    info_df = info_df[info_df.run.isin(runs) & (info_df.feedback == feedback_on)]
    targets = np.array(info_df['target'])
    indexes = list(info_df['trial'] - 1)
    group = list(info_df['run'])

    # load mask
    mask_img = mtk.get_full_mask('/clmnlab/GL/fmri_data/masks/', 'full_mask.group.nii.gz')

    for subj in subj_list:
        # load t-value fMRI images
        img_list = [mtk.load_5d_fmri_image(data_dir + '%s/stats/tvals.%s.r%02d.nii' % (subj, subj, run))
                    for run in runs]

        # check image size
        for img in img_list:
            assert img.shape == (96, 114, 96, 145)

        # indexing image
        imgs = nilearn.image.concat_imgs(img_list)
        imgs = nilearn.image.index_img(imgs, indexes)

        # run searchlight - leave one group out (across run)
        searchlight_img = mtk.run_searchlight(
            mask_img, imgs, targets,
            group=group, group_k=4, radius=radius, estimator=estimator, chance_level=chance_level)

        # save output file
        if feedback_on == 1:
            output_fname = '%s_r%d_fdon.nii.gz' % (subj, radius)
        elif feedback_on == 0:
            output_fname = '%s_r%d_fdoff.nii.gz' % (subj, radius)
        else:
            raise ValueError('feedback_on variable should be 0 or 1, but has %s' % feedback_on)

        searchlight_img.to_filename('%s%s/%s' % (result_dir, label, output_fname))
