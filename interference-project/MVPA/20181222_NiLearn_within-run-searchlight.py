import clmnlab_libs.mvpa_toolkits as mtk
import nilearn.image
import random
import sys


if __name__ == '__main__':
    random.seed(1222)

    if len(sys.argv) >= 2 and sys.argv[1] in {'move', 'plan', 'color'}:
        label = sys.argv[1]
    else:
        raise ValueError('This code need a first label in {move, plan, color}')
        
    if len(sys.argv) >= 3 and sys.argv[2] in {'3', '4', '5'}:
        run = int(sys.argv[2])
    else:
        raise ValueError('This code need a second label in {3, 4, 5}')

    estimator_name = 'svc'
    radius = 8

    # initialize variables
    data_dir = '/clmnlab/IN/MVPA/LSA_tvals/data/'
    mask_dir = '/clmnlab/IN/MVPA/LSS_betas/data/'
    behav_dir = '/clmnlab/IN/MVPA/LSS_betas/behaviors/'
    result_dir = '/clmnlab/IN/MVPA/LSA_tvals/accuracy_map/'
    stats_dir = '/clmnlab/IN/MVPA/LSA_tvals/statistics/'

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
    mask_img = mtk.get_full_mask(mask_dir)

    for subj in subj_list:
        print('starting run %s, %s label' % (subj, label))

        labels = mtk.get_behavior_data(behav_dir, subj, run, label)
        img = nilearn.image.index_img(
                mtk.load_5d_fmri_image(data_dir + 'tvalsLSA.%s.r0%d.nii.gz' % (subj, run)),
                labels['order'] - 1)

        X = img
        y = list(labels['task_type'])
        
        searchlight_img = mtk.run_searchlight(mask_img, X, y, estimator=estimator_name,
                                              cv=mtk.BalancedShuffleSplit(), chance_level=1/3)
        searchlight_img.to_filename(
            result_dir + '%s/%s_%s_r%d_%s_run%d.nii.gz' % (label, subj, label, radius, estimator_name, run))
