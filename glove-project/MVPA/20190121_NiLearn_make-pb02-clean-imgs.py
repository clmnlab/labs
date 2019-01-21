import nilearn.image
import numpy as np


def load_pb02_image(fname, mask_img, condition_index=None, detrend=True, standardize=True):
    img = nilearn.image.load_img(fname)

    if img.shape != (96, 114, 96, 300):
        raise ValueError('Invalid image shape %s, (expect (96, 114, 96, 300) in GL pb02)' % img.shape)

    # exclude the first 3 slices and the last 9 slices - total 288 slices (2 TRs, 12 trials, 12 blocks)
    img = nilearn.image.index_img(img, np.arange(3, 291))

    # apply indexes in condition feedback on or off
    if condition_index is not None:
        img = nilearn.image.index_img(img, condition_index)

    # detrend & standardize after indexing
    if standardize or detrend:
        img = nilearn.image.clean_img(img, detrend=detrend, standardize=standardize)

    # masking group mask
    img = nilearn.masking.unmask(nilearn.masking.apply_mask(imgs=img, mask_img=mask_img), mask_img)

    return img


if __name__ == '__main__':
    # initialize variables
    raw_dir = '/clmnlab/GL/fmri_data/'
    output_dir = '/clmnlab/GL/fmri_data/MVPA/data/pb02_clean/'

    subj_list = [
        'GL03', 'GL04', 'GL05', 'GL06', 'GL07',
        'GL08', 'GL09', 'GL10', 'GL11', 'GL12',
        'GL14', 'GL15', 'GL16', 'GL17', 'GL18',
        'GL19', 'GL20', 'GL21', 'GL22', 'GL24',
        'GL25', 'GL26', 'GL27', 'GL29'
    ]

    runs = [1, 2, 3, 4]

    # feedback on: (True * 2 TRs * 12 trials + False * 2 TRs * 12 trials) * 6 blocks / in each image file
    feedback_on_indexes = np.arange(288*4)[([True] * 24 + [False] * 24) * 6]

    # feedback off: (False * 2 TRs * 12 trials + True * 2 TRs * 12 trials) * 6 blocks / in each image file
    feedback_off_indexes = np.arange(288*4)[([False] * 24 + [True] * 24) * 6]

    full_mask = nilearn.image.load_img('/clmnlab/GL/fmri_data/masks/full_mask.group.nii.gz')

    for subj in subj_list:
        for run in runs:
            # run 1 -> r02, run 2 -> r03, ... in preprocessed files
            img = load_pb02_image(raw_dir + '%s/preprocessed/pb02.%s.r%02d.volreg.nii.gz' % (subj, run+1, subj),
                                  full_mask, feedback_on_indexes)
            img.to_filename(output_dir + '%s_r%02d_fdon.nii.gz' % (subj, run))

            # run 1 -> r02, run 2 -> r03, ... in preprocessed files
            img = load_pb02_image(raw_dir + '%s/preprocessed/pb02.%s.r%02d.volreg.nii.gz' % (subj, run+1, subj),
                                  full_mask, feedback_off_indexes)
            img.to_filename(output_dir + '%s_r%02d_fdoff.nii.gz' % (subj, run))

