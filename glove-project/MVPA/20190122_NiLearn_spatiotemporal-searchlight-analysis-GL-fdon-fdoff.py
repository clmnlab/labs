import clmnlab_libs.mvpa_toolkits as mtk
import clmnlab_libs.gl_toolkits as gtk
import nilearn.image
import sys

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut


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
    data_dir = '/clmnlab/GL/fmri_data/MVPA/data/pb02_clean/'
    behav_dir = '/clmnlab/GL/fmri_data/MVPA/behaviors/'
    result_dir = '/clmnlab/GL/fmri_data/MVPA/searchlight/'

    subj_list = gtk.subj_list

    runs = [1, 2, 3, 4]

    # estimator initialize
    estimator = LinearDiscriminantAnalysis()
    cv = LeaveOneGroupOut()

    indexes, targets, group = gtk.get_behavior_data(feedback_on)
    chance_level = 1 / 4

    # load mask
    mask_img = nilearn.image.load_img('/clmnlab/GL/fmri_data/masks/full_mask.group.nii.gz')

    for subj in subj_list:
        # load spatiotemporal images
        img_list = [nilearn.image.load_img(
            data_dir + '%s_r%02d_fd%s.nii.gz' % (subj, run, 'on' if feedback_on else 'off')) for run in runs]

        # indexing doesn't need, just concatenating
        imgs = nilearn.image.concat_imgs(img_list)

        # reshaping
        imgs = gtk.spatiotemporal_img_reshape(imgs, 288, 2)

        # check image size
        assert imgs.shape == (96, 114, 96, 288, 2)

        # run spatio-temporal searchlight - leave one group out (across run)
        searchlight_img = mtk.run_spatiotemporal_searchlight(
            mask_img, imgs, targets,
            group=group, radius=radius, estimator=estimator, chance_level=chance_level, cv=cv)

        # save output file
        if feedback_on == 1:
            output_fname = '%s_r%d_fdon.nii.gz' % (subj, radius)
        elif feedback_on == 0:
            output_fname = '%s_r%d_fdoff.nii.gz' % (subj, radius)
        else:
            raise ValueError('feedback_on variable should be 0 or 1, but has %s' % feedback_on)

        searchlight_img.to_filename('%sspatiotemporal-%s/%s' % (result_dir, label, output_fname))
