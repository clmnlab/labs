import nilearn.image
import random

from clmnlab_libs.mvpa_toolkits import run_searchlight, get_full_mask, standardize_session_wise


if __name__ == '__main__':
    random.seed(1130)

    estimator = 'gnb'
    radius = 8

    # initialize variables
    data_dir = '/clmnlab/GA/MVPA/LSS_pb02/data/'
    behav_dir = '/clmnlab/GA/MVPA/LSS_pb02/behaviors/'
    result_dir = '/clmnlab/GA/MVPA/LSS_pb02/accuracy_map/'
    stats_dir = '/clmnlab/GA/MVPA/LSS_pb02/statistics/'

    subj_list = [
        'GA01', 'GA02', 'GA05', 'GA07', 'GA08', 'GA11', 'GA12', 'GA13', 'GA14', 'GA15', 'GA18', 'GA19',
        'GB01', 'GB02', 'GB05', 'GB07', 'GB08', 'GB11', 'GB12', 'GB13', 'GB14', 'GB15', 'GB18', 'GB19'
    ]

    num_subj = len(subj_list)
    runs = (1, 2, 3)

    mask_img = get_full_mask('/clmnlab/GA/MVPA/fullmask_GAGB/', 'full_mask.GAGB01to19.nii.gz')

    for subj in subj_list:
        print('starting run %s' % subj)

        labels = list(range(1, 13)) * 8

        img_list = [
            nilearn.image.load_img(data_dir + 'betasLSS.%s.r01.nii.gz' % subj),
            nilearn.image.load_img(data_dir + 'betasLSS.%s.r02.nii.gz' % subj),
            nilearn.image.load_img(data_dir + 'betasLSS.%s.r03.nii.gz' % subj),
        ]
        try:
            img_list = [
                nilearn.image.index_img(standardize_session_wise(img), list(range(1, 97)))
                for img in img_list
            ]
        except IndexError as e:
            print('! index error occurs in %s, skip this subject' % subj)
            print(str(e))
            continue

        X = nilearn.image.concat_imgs(img_list)
        y = labels * 3
        group = [1] * 96 + [2] * 96 + [3] * 96

        searchlight_img = run_searchlight(mask_img, X, y, group,
                                          group_k=3, radius=radius, estimator=estimator, chance_level=1/12)
        searchlight_img.to_filename(result_dir + '%s_r%d.nii.gz' % (subj, radius))
