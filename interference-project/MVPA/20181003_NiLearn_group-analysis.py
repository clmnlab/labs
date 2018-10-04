import numpy as np
import nilearn.image
import nilearn.mass_univariate
import sys


def load_accuracy_maps(filelist, mask):
    results = []
    
    for fname in filelist:
        score = nilearn.image.load_img(fname).get_data()
        results.append(score)

    results = np.ma.array(results, mask=[mask for i in range(len(filelist))])
        
    return results


def run_group_analysis(scores, mask):
    shape = scores[0, :, :, :].shape
    t_img = np.zeros(shape)
    p_img = np.zeros(shape)

    for j in range(shape[0]):
        for k in range(shape[1]):
            for l in range(shape[2]):
                # check voxel is in group mask
                if mask[j, k, l] is True:
                    continue
                    
                # perform permuted OLS
                p_score, t_score, _ = nilearn.mass_univariate.permuted_ols(
                    np.ones((num_subj, 1)),  # one group
                    scores[:, j, k, l].reshape(-1, 1),  # make data (num_subject, data vector)
                    n_perm=1000,
                    two_sided_test=True,
                    n_jobs=8)

                # save results as image
                t_img[j, k, l] = t_score
                p_img[j, k, l] = p_score

            print('%d, %d, ..., ' % (j, k), end='\r')
            
    return t_img, p_img


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] in {'move', 'plan', 'color'}:
        label = sys.argv[1]
    else:
        raise ValueError('This code need a label in {move, plan, color}')
        
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
    mask_array = np.logical_not(mask_img.get_data().astype(np.bool))
    
    scores = load_accuracy_maps(['%s%s_%s_r8_gnb_3class.nii.gz' % (result_dir, s, label) 
                                 for s in subj_list], mask_array)
    
    t_img, p_img = run_group_analysis(scores, mask_array)
    
    t_img = nilearn.image.new_img_like(mask_img, t_img)
    t_img.to_filename(stats_dir + 'group_%s_r8_gnb_3class_tstat.nii.gz' % label)

    p_img = nilearn.image.new_img_like(mask_img, p_img)
    p_img.to_filename(stats_dir + 'group_%s_r8_gnb_3class_pstat.nii.gz' % label)

