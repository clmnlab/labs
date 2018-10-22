import nilearn.image
import random
import sys

from ..mvpa_toolkits import get_behavior_data, load_fmri_image, run_searchlight


if __name__ == '__main__':
    random.seed(1021)
    
    if len(sys.argv) >= 2 and sys.argv[1] in {'move', 'plan', 'color'}:
        label = sys.argv[1]
    else:
        raise ValueError('This code need a label in {move, plan, color}')

    estimator = 'gnb'
    radius = 8

    if len(sys.argv) >= 3:
        for argv in sys.argv[2:]:
            try:
                opt, value = argv.split('=')
                if opt == 'estimator':
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
                                 + 'estimator=estimator name (gnb or svc)')
        
    # initialize variables
    data_dir = '/clmnlab/IN/MVPA/LSS_betas/data/'
    behav_dir = '/clmnlab/IN/MVPA/LSS_betas/behaviors/'
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
    
    run_number_dict = {
        'move': [3, 5],
        'plan': [3, 4],
        'color': [3, 4],
    }
    
    runs = run_number_dict[label]

    # load mask file
    mask_path = data_dir + 'full_mask.group33.nii.gz'
    mask_img = nilearn.image.load_img(mask_path)

    for subj in subj_list:
        print('starting run %s, %s label' % (subj, label))
        
        for run in runs:
            # load behavioral data
            labels = get_behavior_data(behav_dir, subj, run, label, stratified_group=True)

            # load fmri file
            img = load_fmri_image(data_dir, subj, run, labels)

            X = img
            y = list(labels['task_type'])
            group = list(labels['group'])
            
            searchlight_img = run_searchlight(mask_img, X, y, group, estimator)
            searchlight_img.to_filename(result_dir + '%s_run%d_%s_r%d_%s.nii.gz' % (subj, run, label, radius, estimator))