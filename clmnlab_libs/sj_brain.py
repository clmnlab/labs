
# Common Libraries
import os
import numpy as np
import pandas as pd
import copy
import datetime

# Preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from scipy.stats import zscore

# RSAtoolbox
import rsatoolbox
from rsatoolbox.inference import eval_fixed
from rsatoolbox.model import ModelFixed
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs, evaluate_models_searchlight

# Visualize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

# Brain
import nilearn
import nibabel as nb
from nilearn import image
from nilearn.plotting import plot_roi, plot_design_matrix
from nilearn.image import resample_to_img, math_img
import nltools
from nltools.data import Brain_Data

# Custom Libraries
import sj_util
import sj_sequence
import sj_file_system
import sj_higher_function
import sj_preprocessing
import sj_file_system

# Sources

def load_behaviors(behavior_paths):
    """
    load behaviors
    
    :param behavior_paths: behavior data paths(string)
    
    return behavior data list(element - dataframe)
    """
    behaviors = []
    for i in range(0, len(behavior_paths)):
        behaviors.append(pd.read_csv(behavior_paths[i]))
    return behaviors

def load_head_motion_datas(head_motion_paths):
    """
    load head motion datas from paths
    
    :param head_motion_paths: head motion data paths(string)
    
    return headmotion data list(element - dataframe)
    """
    add_reg_names = ["tx", "ty", "tz", "rx", "ry", "rz"]

    head_motion_datas = []
    for data_path in head_motion_paths:
        head_motion_data = pd.read_csv(data_path, delimiter=" ", names=add_reg_names)
        head_motion_datas.append(head_motion_data)

    return head_motion_datas

def type_transform(brain_data, tranform_type):
    """
    Transform fmri data type 
    
    :param brain_data: fmri_data (Nifti1Image, nltools Brain_data)
    :param tranform_type: nltools, array, nibabel
    return transformed data
    """
    if type(brain_data) == nb.Nifti1Image:
        if tranform_type == "nltools":
            result = nltools.data.brain_data.Brain_Data(brain_data)
        elif tranform_type == "array":
            result = brain_data.get_fdata()
    elif type(brain_data) == nltools.data.brain_data.Brain_Data:
        if tranform_type == "array":
            result = brain_data.to_nifti().get_fdata()
        elif tranform_type == "nibabel":
            result = brain_data.to_nifti()
            
    return result

def nifiti_4d_to_1d(img):
    """
    transform image shape from 4d to 1d
    
    slice image per time and flatten

    :param img: nifti image
    
    return: array(element: 1d array) / shape (n x p) / n: measure_timing, p: voxels
    """
    results = []

    time = img.shape[-1]
    for t in range(0, time):
        sliced_img = img.slicer[..., t]
        array1d = sliced_img.get_fdata().reshape(-1)
        results.append(array1d)
    return np.array(results)

def concat_fMRI_datas(interest_fMRI_data_indexes = None, fMRI_datas = []):
    """
    concatenate fMRI datas by flattening

    :params interest_fMRI_data_indexes: fMRI_datas's indexes ex) [1,2,3]
    :params fMRI_datas: array of Nifti1Image

    return: flatten fMRI data(3d numpy array)
    """

    if interest_fMRI_data_indexes == None:
        array_fMRI_datas = []
        for interest in range(0, len(fMRI_datas)):
            fMRI_data = fMRI_datas[interest]
            array_fMRI_datas.append(nifiti_4d_to_1d(fMRI_data))
        interest_fMRI_data = np.concatenate(array_fMRI_datas, axis=0)
    else:
        array_fMRI_datas = []
        for interest in interest_fMRI_data_indexes:
            fMRI_data = fMRI_datas[interest]
            array_fMRI_datas.append(nifiti_4d_to_1d(fMRI_data))
        interest_fMRI_data = np.concatenate(array_fMRI_datas, axis=0)
    return interest_fMRI_data

def apply_mask_change_shape(fMRI_datas, mask):
    """
    Convert fMRI data(3d) applied mask to 1d data
    
    :param fMRI_datas: fMRI_datas(list - nitfti image)
    :param mask: nifti_mask
    
    return masked_fmri_datas(1d array), mask_img
    """
    resampled_mask = resample_to_img(mask, fMRI_datas[0], interpolation="nearest")
    resampled_mask = nb.Nifti1Image(np.array(resampled_mask.get_fdata() > 0, dtype=np.int8),
                                         resampled_mask.affine)
    
    # Multiply the functional image with the mask
    roi_fMRI_datas = []
    for target_data in fMRI_datas:
        roi_img = nilearn.masking.apply_mask(target_data, resampled_mask)
        roi_fMRI_datas.append(roi_img)
        
    return roi_fMRI_datas, resampled_mask

def apply_mask_no_change_shape(fMRI_datas, mask):
    """
    While preserving shape, apply ROI mask to fMRI datas 
    
    :params fMRI_datas: array of Nifti1Image(list)
    :params mask: Nifti1Image (no need to fit shape to fMRI_datas)
    
    return fMRI_dats(list), mask_img
    """
    resampled_mask = resample_to_img(mask, fMRI_datas[0], interpolation="nearest")
    resampled_mask = nb.Nifti1Image(np.array(resampled_mask.get_fdata() > 0, dtype=np.int8),
                                         resampled_mask.affine)

    # Multiply the functional image with the mask
    roi_fMRI_datas = []
    for target_data in fMRI_datas:
        if len(target_data.shape) == len(resampled_mask.shape):
            roi_img = math_img('img1 * img2', img1=target_data, img2=resampled_mask)
            roi_fMRI_datas.append(roi_img)
        else:
            roi_img = math_img('img1 * img2', img1=target_data, img2=resampled_mask.slicer[..., None])
            roi_fMRI_datas.append(roi_img)

    return roi_fMRI_datas, resampled_mask

def apply_mask(fMRI_datas, mask):
    """
    While removing zeros, apply ROI mask to fMRI datas
    
    :params fMRI_datas: array of Nifti1Image(list)
    :params mask: Nifti1Image (no need to fit shape to fMRI_datas)
    
    return fMRI_dats(list), mask_img
    """
    roi_datas, resampled_mask = apply_mask_no_change_shape(fMRI_datas, mask)

    # Remove assplit_data_pairs many zero rows in the data matrix to reduce overall volume size
    from nilearn.image import crop_img

    roi_crop_fMRI_datas = []
    for roi_data in roi_datas:
        roi_crop_fMRI_datas.append(crop_img(roi_data))
    return roi_crop_fMRI_datas, resampled_mask

def apply_mask_with_img(anatomy_data, fMRI_datas, mask, is_show_img = True):
    """
    Applying mask while showing the result of mask application
    
    :params anatomy_data: anatomy(nifiti1Image)
    :params fMRI_datas: array of Nifti1Image(list)
    :params mask: Nifti1Image (no need to fit shape to fMRI_datas)
    
    return fMRI_dats(list), mask_img
    """
    if len(fMRI_datas[0].shape) == 3:
        reference_img = fMRI_datas[0]
    else:
        reference_img = image.mean_img(fMRI_datas[0])
    resampled_anatomy = image.resample_to_img(anatomy_data, reference_img)
    resampled_mask = resample_to_img(mask, fMRI_datas[0], interpolation="nearest")
    
    if is_show_img:
        plot_roi(resampled_mask,
                 black_bg=True,
                 bg_img=resampled_anatomy,
                 cut_coords=None,
                 cmap='magma_r',
                 dim=1)
    return apply_mask(fMRI_datas, mask)

def flatten(fMRI_datas):
    """
    flatten voxel shape

    :param fMRI_datas: array of Nifti1Image
    
    return flattened fMRI datas(list - numpy array)
    """
    flatten_datas = []
    for data in fMRI_datas:
        count_of_timing = data.shape[-1]
        flatten_datas.append(data.get_fdata().reshape(count_of_timing, -1))
    return flatten_datas

def split_data_pairs(datas, behaviors, train_indexes, test_indexes):
    """
    Split fMRI datas and behavior datas to obtain train and test dataset
    
    :param datas: fmri_datas(list - nifti_image) - Each image is separted by run probably
    :param behaviors: behaviors(list - dataframe) - Each behavior is separted by run probably
    :param train_indexes: Train data set indexes, Each index probably represents a run
    :param test_indexes: Test data set indexes, Each index probably represents a run
    
    return train_datas, train_behavior, test_datas(4d numpy array), test_behavior
    """
    train_datas = concat_fMRI_datas(train_indexes, datas)
    train_behavior = sj_util.get_multiple_elements_in_list(in_list=behaviors,
                                                           in_indices=train_indexes)
    train_behavior = sj_util.concat_pandas_datas(train_behavior)

    test_datas = concat_fMRI_datas(test_indexes, datas)
    test_behavior = sj_util.get_multiple_elements_in_list(in_list=behaviors,
                                                          in_indices=test_indexes)
    test_behavior = sj_util.concat_pandas_datas(test_behavior)

    return train_datas, train_behavior, test_datas, test_behavior

def get_specific_images(img, mask_condition):
    result = nilearn.image.index_img(img, list(map(lambda x: x[0], np.argwhere(mask_condition))))

    return result


def highlight_stat(roi_array, stat_array, stat_threshold):
    """
    highlight roi area's statistics in stat map,

    :param roi_array: array
    :param stat_array: array
    :param stat_threshold: threshold
    """
    highlight_array = roi_array.copy()
    highlight_array[:] = 1

    non_highlight_array = highlight_array.copy()
    non_highlight_array[:] = -1

    zero_array = non_highlight_array.copy()
    zero_array[:] = 0

    conditions = [np.logical_and(roi_array > 0, stat_array > stat_threshold), stat_array > stat_threshold, True]

    from matplotlib import cm
    color_map = cm.get_cmap('viridis', 2)

    return_obj = {
        "data": np.select(conditions, [highlight_array, non_highlight_array, zero_array]),
        "color_map": color_map
    }

    return return_obj


def colored_roi_with_stat(roi_arrays, stat_map, stat_threshold):
    """
    show colorred roi and represent stat

    :param roi_arrays: array of roi
    :param stat_map: array of statistics
    :param stat_threshold: threshold
    """
    # preprocessing
    roi_arrays = [roi.astype(np.int16) for roi in roi_arrays]

    # pre-data
    shape = roi_arrays[0].shape

    # make roi array
    zero_array = np.repeat(0, shape[0] * shape[1] * shape[2]).reshape(shape)
    zero_array[:] = 0

    color_data = 1
    colored_roi_arrays = [roi.copy() for roi in roi_arrays]
    for roi in colored_roi_arrays:
        roi[roi == True] = color_data
        color_data += 1

    # make roi_stat hightlight
    color_data += 1
    roi_stat_highlight = []
    color_values = []
    conditions = []
    for roi in roi_arrays:
        conditions += [np.logical_and(stat_map > stat_threshold, roi == True)]

        color_values.append(color_data)
        roi_stat_highlight.append(np.repeat(color_data, shape[0] * shape[1] * shape[2]).reshape(shape))

    # conditions
    conditions += [stat_map > stat_threshold]  # highlight
    conditions += [roi > 0 for roi in roi_arrays]
    conditions += [True]

    color_data += 1
    stat_array = np.repeat(color_data, shape[0] * shape[1] * shape[2]).reshape(shape)

    result = np.select(conditions, roi_stat_highlight + [stat_array] + colored_roi_arrays + [zero_array])

    from matplotlib import cm
    color_map = cm.get_cmap('viridis', len(result) - 1)  # -1: remove zero

    return_obj = {
        "data": result,
        "color_map": color_map
    }

    return return_obj

def mean_img(imgs, threshold=None):
    """
    mean nilearn images
    
    :param imgs: target images to mean(nilearn image 4d)
    :papram threshold: if this value is set, mean_img is subtracted by this value
    """
    result = nilearn.image.mean_img(imgs)
    
    if threshold != None:
        result = nb.Nifti1Image(result.get_fdata() - threshold, affine = result.affine)
    return result

def add_imgs(imgs, is_use_path = False):
    """
    Add many image from imgs

    :param imgs: imgs(nitfti image array)

    return nifti image
    """
    
    if len(imgs) == 1:
        if is_use_path == True:
            img = nb.load(imgs[0])
            return img
        else:
            return imgs[0]
    
    if is_use_path == True:
        temp_imgs = []
        for i in range(len(imgs)):
            img = nb.load(imgs[i])
            temp_imgs.append(img)
        imgs = temp_imgs
        
    return math_img("img1 + img2", img1=imgs[0], img2=add_imgs(imgs[1:]))

def join_roi_imgs(imgs, is_use_path = False):
    """
    join roi images

    :param imgs: array(nifti images) / nitfti image must have binary element

    return roi image(nifti)
    """
    if is_use_path == True:
        img = nb.load(imgs[0])
        if len(imgs) == 1:
            return img

        return math_img("img1 * img2", img1=img, img2=join_roi_imgs(imgs[1:]))
    else:
        if len(imgs) == 1:
            return imgs[0]
        return math_img("img1 * img2", img1=imgs[0], img2=join_roi_imgs(imgs[1:]))

def load_mask(mask_path, resample_target):
    resampled_mask = resample_to_img(nb.load(mask_path), resample_target, interpolation="nearest")
    return resampled_mask

def mean_img_within_diff(fMRI_data, lower_diff, upper_diff):
    """
    mean image from fMRI_data within target_lower_bound <= ~ <= target_upper_bound

    :param fMRI_data: Niimg-like obj
    :param lower_diff: lower bound about each index, unsigned integer
    :param upper_diff: upper bound about each index, unsigned integer
    """
    fmri_data_count = fMRI_data.shape[-1]

    mean_datas = []
    for target_index in range(0, fmri_data_count):
        # get data from fMRI_data within target_lower_bound <= ~ <= target_upper_bound
        target_lower_bound = target_index - lower_diff
        target_upper_bound = target_index + upper_diff + 1

        if target_lower_bound < 0:
            target_lower_bound = 0
        if target_upper_bound > fmri_data_count:
            target_upper_bound = fmri_data_count

        # slice data
        sliced_fMRI_data = fMRI_data.slicer[..., target_lower_bound: target_upper_bound]

        # mean
        mean_data = image.mean_img(sliced_fMRI_data)
        mean_datas.append(mean_data)

    return image.concat_imgs(mean_datas)

def mean_img_within_diff_with_targetIndex(fMRI_data, lower_diff, upper_diff, target_indexes):
    """
    mean image from fMRI_data within target_lower_bound <= ~ <= target_upper_bound

    :param fMRI_data: Niimg-like obj
    :param lower_diff: lower bound about each index, unsigned integer
    :param upper_diff: upper bound about each index, unsigned integer
    :param target_indexes: interest indexes ex) [1,2,3]
    """
    fmri_data_count = fMRI_data.shape[-1]

    mean_datas = []
    for target_index in target_indexes:
        # get data from fMRI_data within target_lower_bound <= ~ <= target_upper_bound
        target_lower_bound = target_index - lower_diff
        target_upper_bound = target_index + upper_diff + 1

        if target_lower_bound < 0:
            target_lower_bound = 0
        if target_upper_bound > fmri_data_count:
            target_upper_bound = fmri_data_count

        #format = "target: {target_index} lower: {lower}, upper: {upper}"
        #print(format.format(target_index=target_index, lower=target_lower_bound, upper=target_upper_bound))

        # slice data
        sliced_fMRI_data = fMRI_data.slicer[..., target_lower_bound: target_upper_bound]

        # mean
        mean_data = image.mean_img(sliced_fMRI_data)
        mean_datas.append(mean_data)

    return image.concat_imgs(mean_datas)

def upper_tri(RDM):
    """
    upper_tri returns the upper triangular index of an RDM
    
    :param RDM: squareform RDM(numpy array)
    
    return upper triangular vector of the RDM(1D array) 
    """
    
    # returns the upper triangle
    m = RDM.shape[0]
    r, c = np.triu_indices(m, 1)
    return RDM[r, c]

def total_RDM_dissimilarity(RDM):
    """
    total dissimilarity 
    
    :param RDM: RDM(numpy 2d array)
    
    return: scalar value
    """
    
    # The reason excluding zeron is to exclude non-calculated area(ex: external area of mask)
    up_RDM = upper_tri(RDM)
    return np.mean(up_RDM[up_RDM != 0])
    
def searchlight_RDM(betas, 
                    mask, 
                    conditions, 
                    save_rdm_path = None, 
                    save_region_path = None,
                    radius=2, 
                    threshold=1,
                    method="correlation"):
    """
    Searches through the non-zero voxels of the mask, selects centers where
    proportion of sphere voxels >= self.threshold
    
    threshold (float, optional): Threshold of the proportion of voxels that need to
        be inside the brain mask in order for it to be
        considered a good searchlight center.
        Values go between 0.0 - 1.0 where 1.0 means that
        100% of the voxels need to be inside
        the brain mask.
        Defaults to 1.0.
    
    Each beta value's element must match with corresponding condition
    
    if you want to use crossvalidation RDM then many beta values need to be insulted.
    Crossvalidation RDM works using default cv descriptor.
    
    Pattern descriptor's index is allocated by the order of conditions excluding conditions already mapped to index.
    
    :param betas: beta values(nifti array or nltools Brain data)
    :param mask: must be binary data
    :param conditions: conditions(list of string)
    :param save_region_path: if save_rdm_path is not none, save region data
    :param save_rdm_path: if save_rdm_path is not none, save rdm data
    :param radius: searchlight radius
    :param threshold: threshold(float)
    :param method: distance method
    
    return RDMs(rsatoolbox)
    """
    
    if save_rdm_path != None:
        try:
            print("load rdm: ", save_rdm_path)
            SL_RDM = sj_file_system.load(save_rdm_path)
            return SL_RDM
        except:
            print("load rdm fail: ", save_rdm_path)
            
    print("RDM Calculate Start")    
    if type(betas) == list and type(betas[0]) == nb.Nifti1Image:
        # checking shape is same
        assert all(map(lambda data: data.shape == betas[0].shape, betas)), "nifti_datas element shape is not same"

        array_betas = np.array([betas[data_i].get_fdata() for data_i in range(0, len(betas))])
    elif type(betas) == nb.Nifti1Image:
        array_betas = []
        for i in range(len(conditions)):
            beta = betas.slicer[..., i]
            array_betas.append(beta.get_fdata())
        array_betas = np.array(array_betas)
    elif type(betas) == nltools.data.brain_data.Brain_Data:
        array_betas = []
        for condition_i in range(0, len(conditions)):
            array_betas.append(betas[condition_i].to_nifti().get_fdata())    

        array_betas = np.array(array_betas)
    elif type(betas) == list and type(betas[0]) == nltools.data.brain_data.Brain_Data:
        array_betas = np.array([beta.to_nifti().get_fdata() for beta in betas])
    
    if save_region_path != None:        
        try:
            centers, neighbors = sj_file_system.load(save_region_path)
        except:
            print("load region fail: ", save_region_path)
            """
            Searches through the non-zero voxels of the mask, 
            selects centers where proportion of sphere voxels >= self.threshold
            
            This process searches neighbors matched within radius using euclidean distance.
            
            reference: 
            https://rsatoolbox.readthedocs.io/en/latest/_modules/rsatoolbox/util/searchlight.html#get_volume_searchlight
            
            note!!!!!!!!
            RDM searchlight uses an index and calculates the Euclidean distance to apply a mask.(not mm)
            
            """
            centers, neighbors = get_volume_searchlight(mask.get_fdata(), 
                                                    radius=radius, 
                                                    threshold=threshold)
            sj_file_system.save((centers, neighbors), save_region_path)
            print("save region: ", save_region_path)
        
    # reshape data so we have n_observastions x n_voxels
    n_conditions, nx, ny, nz = array_betas.shape
    print(n_conditions, nx, ny, nz)
    
    data_2d = array_betas.reshape([n_conditions, -1])
    print(data_2d.shape)
    print(len(conditions))
    data_2d = np.nan_to_num(data_2d)
    
    """
    reference: 
    """
    SL_RDM = get_searchlight_RDMs(data_2d=data_2d, 
                                  centers=centers, 
                                  neighbors=neighbors, 
                                  events=conditions, 
                                  method=method)
    
    if save_rdm_path != None:
        print("save RDM: ", save_rdm_path)
        sj_file_system.save(SL_RDM, save_rdm_path)
        
    return SL_RDM
                  
def RSA(models, 
        mask,
        save_rdm_path = None,
        save_corr_brain_path = None,
        datas = None, 
        conditions = None,
        region_path = None,
        radius = 2,
        threshold = 1,
        n_jobs=1,
        rdm_distance_method="correlation",
        debug=None):
    """
    Do representational Similarity Analysis
    
    Searches through the non-zero voxels of the mask, selects centers where
    proportion of sphere voxels >= self.threshold
    
    threshold (float, optional): Threshold of the proportion of voxels that need to
        be inside the brain mask in order for it to be
        considered a good searchlight center.
        Values go between 0.0 - 1.0 where 1.0 means that
        100% of the voxels need to be inside
        the brain mask.
        Defaults to 1.0.
        
    :param models: RDM Model list, made by RDM_model
    :param mask: must be binary data
    :param save_rdm_path: (string) if save_rdm_path is not none, save rdm data
    :param save_corr_brain_path: (string) correlation brain with model
    :param datas: nifti array or nltools Brain data
    :param conditions: data conditions(1d list)
    :param region_path: (string) if region_path is not none, save region data
    :param radius: searchlight radius
    :param threshold: threshold(float)
    :param rdm_distance_method: distance method
    
    return: RDM_brains
    """
    # input validation
    assert ((type(datas) != None and type(conditions) != None) or load_rdm_path != None), "Please input nifti data or load_rdm_path"
    
    # model validation
    for model in models:
        # check all model's condition is same
        assert model.conditions == models[0].conditions, "all models condition is not matched!!"
        
        # check model degree of freedom for computing correlation
        assert len(np.unique(upper_tri(model.model))) != 1, "degree of freedom is 0!"
    
    # Calculate RDM
    SL_RDM = searchlight_RDM(betas=datas, 
                 conditions=conditions,
                 mask=mask,
                 save_rdm_path=save_rdm_path,
                             save_region_path = region_path,
                 radius=radius,
                 threshold=threshold,
                 method=rdm_distance_method)
    
    if debug == 1:
        return SL_RDM
    
    # Get Correlation between RDM and model
    fixed_models = [ModelFixed("", upper_tri(model.model)) for model in models]
    eval_results = evaluate_models_searchlight(SL_RDM,
                                               fixed_models,
                                               eval_fixed,
                                               method='spearman',
                                               n_jobs=n_jobs)
    
    eval_results = [e.evaluations for e in eval_results]
    
    eval_score = np.array(list(map(lambda x: x.reshape(-1), eval_results)))
    scores = np.array(list(map(lambda score: score.reshape(-1), eval_score)))
    
    if debug == 2:
        return scores
    
    # Make RDM Brains
    corr_brains = []
    for model_index in range(0, len(models)):
        # Create an 3D array, with the size of mask, and 
        x, y, z = mask.shape
        corr_brain = np.zeros([x*y*z])
        corr_brain[list(SL_RDM.rdm_descriptors['voxel_index'])] = scores[:,model_index]
        corr_brain = corr_brain.reshape([x, y, z])
        corr_brains.append(corr_brain)
    
    # save corr which contains brain RDM with model
    if save_corr_brain_path != None:
        for model_i in range(0, len(models)):
            model = models[model_i]
            
            save_corr_path = sj_file_system.str_join([save_corr_brain_path, "corr", model.name]) + ".nii.gz"
            
            corr_img = nb.Nifti1Image(corr_brains[model_i], affine = mask.affine)
            nb.save(corr_img, save_corr_path)  
    
    # result
    result = {}
    for model_i in range(0, len(models)):
        model = models[model_i]
        result[model.name] = corr_brains[model_i]
            
    return result

class RDM_model:
    """
    This class's purpose is managing RDM model
    """
    
    def __init__(self, model_2d_array, model_name, conditions):
        """
        :param model_2d_array: model(2d numpy array)
        :param model_name: model name(str)
        :param conditions: conditions(list of string)
        """
        self.model = model_2d_array
        self.name = model_name
        self.conditions = conditions
        
    def draw(self, fig = None, axis = None, cmap="rainbow", v_range = (0, 1)):
        if fig is None and axis is None:
            fig, axis = plt.subplots(1,1)
        
        RDM_model.draw_rdm(rdm=self.model, 
                           conditions=self.conditions, 
                           fig = fig,
                           title=self.name, 
                           cmap=cmap,
                           axis = axis,
                           v_range = v_range)
    
    # Utility function
    @staticmethod
    def draw_rdm(rdm, 
                 conditions, 
                 fig,
                 axis,
                 title="", 
                 cmap="rainbow",
                 v_range = (0, 1)):
        """
        :param rdm: numpy 2d array
        :param conditions: list of condition
        :param y_range: (y_min, y_max)
        :param axis: axis
        :param v_range: color ranging ex) (0, 1)
        """
        ticks_range = np.arange(0, len(conditions))
        
        divider = make_axes_locatable(axis)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = axis.imshow(rdm, cmap="rainbow", vmin = v_range[0], vmax = v_range[1])
        fig.colorbar(im, cax=cax, orientation='vertical')
        axis.set_xticks(ticks_range, conditions, rotation = 90)
        axis.set_yticks(ticks_range, conditions)
        axis.set_title(title)
        
def make_RDM_brain(brain_shape, RDMs, conditions, is_return_1d=False):
    """
    Make RDM brain (nx, ny, nz, n_condition x n_condition)
    
    :param brain_shape: x,y,z(tuple)
    :param RDMs: rsatoolbox.rdm.RDMs
    
    return 5d array(x, y, z, condition, condition)
    """
    assert type(RDMs) == rsatoolbox.rdm.RDMs, "Please input rsatoolbox RDMs"
    
    condition_length = len(conditions)
    
    x, y, z = brain_shape
    
    brain_1d = list(np.zeros([x * y * z]))
    brain_1d_RDM = list(map(lambda _: np.repeat(0, condition_length * condition_length).reshape(condition_length, 
                                                                                                condition_length).tolist(), 
                        brain_1d))
    
    for RDM in RDMs:
        voxel_index = RDM.rdm_descriptors["voxel_index"]
        rdm_mat = RDM.get_matrices()
    
        assert len(voxel_index) == 1 and len(rdm_mat) == 1, "multi voxel index is occured"
        
        voxel_index = voxel_index[0]
        rdm_mat = rdm_mat[0]
        
        brain_1d_RDM[voxel_index] = rdm_mat.tolist()
    
    if is_return_1d:
        return brain_1d_RDM
    else:
        return np.array(brain_1d_RDM).reshape([x, y, z, condition_length, condition_length])
    
    return brain_1d_RDM

def brain_total_dissimilarity(rdm_brain):
    """
    Get total dissimilarity from rdm_brain
    
    :param rdm_brain: 5d array(array) made by make_RDM_brain function
    
    return np.array(nx, ny, nz)
    """
    nx, ny, nz = rdm_brain.shape[0], rdm_brain.shape[1], rdm_brain.shape[2]

    result = np.zeros([nx,ny,nz])
    for i in range(nx):
        for j in range(ny):
            for z in range(nz):
                result[i][j][z] = total_RDM_dissimilarity(rdm_brain[i][j][z])
    return result

def masked_rdm_brain(rdm_brain, nifti_mask, debug=None):
    """
    Apply mask to RDM brain
    
    :param rdm_brain: 5d array(array)
    :param nifti_mask: nifti
    
    return masked_data_only
    """
    rdm_shape = rdm_brain.shape
    rdm_brain_1d = rdm_brain.reshape(-1, rdm_shape[3], rdm_shape[4])
    mask_data_1d = nifti_mask.get_fdata().reshape(-1)
    
    if debug != None:
        # rdm_brain과 nifti mask를 1차원으로 축약해서 mask를 씌워도
        # 같다는 공간이라는 것을 보이기 위함
        test = np.sum(np.sum(rdm_brain_1d, axis=1), axis = 1)
        
        for i in range(0, len(mask_data_1d)):
            if mask_data_1d[i] == True:
                test[i] = np.sum(test[i])
            else:
                test[i] = 0
        return test
    
    masked_data_only = rdm_brain_1d[mask_data_1d > 0, :, :]
    
    return masked_data_only

def make_roi(roi_paths, reference_img):
    """
    join roi images

    :params roi_paths: roi path(nifiti img)
    :params reference_img: reference_img for fitting shape

    return: roi(nifiti img)
    """
    roi = add_imgs(roi_paths)
    roi = image.resample_to_img(roi, reference_img, interpolation="nearest")
    
    return roi
        
class fan_roi_mask_manager:
    def __init__(self, fan_info_path, mask_dir_path, reference_img):
        """
        :params fan_info_path: fan roi path
        :params reference_img: reference_img for fitting shape
        """
        self.reference_img = reference_img
        
        # load fan info
        self.mask_fan_info = pd.read_csv(fan_info_path, header=None)
        self.mask_fan_info.index = self.mask_fan_info.index + 1 # roi file_name is started from 1
        self.mask_fan_info.index = list(map(lambda index: str(index).zfill(3), self.mask_fan_info.index)) # for matching filename format ex) 001
        self.mask_fan_info.columns = ["Description"]
        self.mask_fan_info = dict(self.mask_fan_info["Description"])
    
        self.mask_dir_path = mask_dir_path
        
    def search_mask_info(self, keywords):
        """
        search roi information using keywords
        
        :param keywords: keyword for searching ex) ["Rt", "prefrontal"]
        """
        searched_dict = sj_sequence.search_dict(self.mask_fan_info, keywords)
        return searched_dict
    
    def search_roi(self, keywords):
        """
        :param keywords: keyword to search(list)
        """
        return brain_mask(mask_nifti_img = self.make_roi_with_search(keywords = keywords),
                          name=sj_file_system.str_join(keywords))
    
    def make_roi_with_search(self, keywords):
        """
        search roi paths and return roi(nifiti img)

        :params dict_roi_info: roi info dictionary
        :params keywords: search keywords ex ["precentral gyrus", "subiculum"]
        :params reference_img: reference_img for fitting shape

        return: roi(nifiti img)
        """
        searched_dict = sj_sequence.search_dict(self.mask_fan_info, keywords)

        search_paths = [os.path.join(self.mask_dir_path, "fan.roi." + key + ".nii.gz") for key in searched_dict] 

        return make_roi(search_paths, reference_img = self.reference_img)

class parcellation_roi_mask_manager:
    def __init__(self, parcellation_info_path, mask_path, reference_img):
        """
        :params parcellation_info_path: parcellation roi info path
        :params mask_path: mask path
        :params reference_img: reference_img for fitting shape
        """
        self.mask_info = self.load_roi_info(parcellation_info_path)
        self.mask_path = mask_path
        self.reference_img = reference_img
        
    def load_roi_info(self, parcellation_info_path):
        """
        load roi information
        
        :params parcellation_info_path: parcellation roi info path
        return { roi description : index }
        """
        with open(parcellation_info_path) as f:
            lines = f.readlines()
        
        parcellation_lines = sj_higher_function.list_map(lines[1:], lambda s: s.strip())
        parcellation_lines = sj_higher_function.list_map(parcellation_lines, lambda s: s.split("="))
        
        parcellation_info = {}
        for key, description in parcellation_lines:
            parcellation_info[int(key)] = description
        
        return parcellation_info
        
    def search_mask_info(self, keywords):
        """
        search roi information using keywords
        
        :param keywords: keyword for searching ex) ["Rt", "prefrontal"]
        """
        searched_dict = sj_sequence.search_dict(self.mask_info, keywords)
        return searched_dict
    
    def make_roi_with_search(self, 
                             keywords):
        """
        search roi paths and return roi(nifiti img)
        
        :params mask_path: mask path
        :params dict_roi_info: roi info dictionary
        :params keywords: search keywords ex ["precentral gyrus", "subiculum"]
        :params reference_img: reference_img for fitting shape

        return: roi(nifiti img)
        """
        logic_to_number = np.vectorize(lambda x: 1 if x == True else 0)
        
        searched_dict = sj_sequence.search_dict(self.mask_info, keywords)
        
        parcellation_mask = Brain_Data(self.mask_path)
        parcellation_mask_x = nltools.mask.expand_mask(parcellation_mask)

        local_masks = []
        for key in searched_dict:
            local_mask = parcellation_mask_x[key].to_nifti()
            local_mask_array = logic_to_number(local_mask.get_fdata())

            local_mask = nb.Nifti1Image(logic_to_number(local_mask.get_fdata()), local_mask.affine)

            local_masks.append(local_mask)
    
        roi_img = add_imgs(local_masks)
        roi_img = image.resample_to_img(roi_img, self.reference_img, interpolation="nearest")

        return roi_img
    
class brain_mask:
    def __init__(self, mask_nifti_img, name):
        """
        :param mask_nifti_img: (nifti image) or path
        :param name: mask name(string)
        """
        if type(mask_nifti_img) == str:
            self.mask_nifti_img = nb.load(mask_nifti_img)
        elif type(mask_nifti_img) == nb.Nifti1Image:
            self.mask_nifti_img = mask_nifti_img
        else:
            raise ValueError('mask_nifti_img need to be set nifti image file')
            
        self.name = name
       
    def get_data(self):
        return self.mask_nifti_img
    
    def apply(self, anatomy, fMRI_datas, is_show_img = True):
        """
        Apply Mask to fMRI datas
        
        :param anatomy: for viewing background
        :param fMRI_datas: nifti(list)
        :param is_show_img: True shows the roi image
        
        return fmri_datas applied mask (these datas' shape is changed by mask)
        """
        
        image_only = 0
        return apply_mask_with_img(anatomy_data = anatomy,
                                   fMRI_datas = fMRI_datas, 
                                   mask = self.mask_nifti_img,
                                   is_show_img = is_show_img)[image_only]
    
def construct_contrast(design_matrix_columns, contrast_info):
    """
    construct contrast array
    
    :param design_matrix_columns: columns(list)
    :param contrast_info: dictionary
    
    return contrast array
    
    example)
    construct_contrast(["['1', '4', '2', '3', '1', '2', '4', '3']"], {"['1', '4', '2', '3', '1', '2', '4', '3']" : 1})
    
    """
    
    assert sj_sequence.check_duplication(design_matrix_columns) != True, "column is duplicated!"

    candidates = np.zeros(len(design_matrix_columns))
    for condition in contrast_info:
        i_condition = design_matrix_columns.index(condition)
        candidates[i_condition] = contrast_info[condition]

    return candidates

def plot_timeseries(axis, data, showing_x_interval = 10, labels=None, title = None, ylim = None, linewidth=3):
    """
    Plotting data as time series
    
    :params axis: plt axis
    :params data: (np.ndarray) signal varying over time, where each column is a different signal.
    :params showing_x_interval: x-axis interval to show in axis
    :params labels: (list) labels which need to correspond to the number of columns.
    :params title: (string) title of plot
    :params ylim: y_lower, y_upper(tuple) limitation where the y-axis is limited 
    :params linewidth: line width which plot draws line as
    """
    axis.plot(data, linewidth=linewidth)
    axis.set_ylabel('Intensity', fontsize=18)
    axis.set_xlabel('MRI_Time', fontsize=18)
    axis.set_xticks(np.arange(0, len(data), 10))
    axis.tick_params(axis='x', rotation=90)
    
    if ylim != None:
        axis.set_ylim(ylim)
        
    if title != None:
        axis.set_title(title)
        
    if labels is not None:
        if len(labels) != data.shape[1]:
            raise ValueError('Need to have the same number of labels as columns in data.')
        axis.legend(labels, fontsize=18, loc="upper right")

def VIF(design_matrix):
    """
    Calculate VIF from design matrix
    
    return vif(dataframe)
    """
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = design_matrix.columns
    vif_data["VIF"] = [variance_inflation_factor(design_matrix.values, i) for i in range(len(design_matrix.columns))]

    return vif_data

def get_statsWithDM(fMRI_datas, 
                    dsg_mats, 
                    full_mask):
    """
    get stat values from dsg_mats
    
    :param fMRI_datas: fMRI_datas(list of nilearn)
    :param dsg_mats: design matrixes
    :param full_mask: full_mask
    
    return: stats, dsg_mats
    """
    # convert nilearn -> nltools
    brain_datas = [Brain_Data(data, mask=full_mask) for data in fMRI_datas]
    for run_index in range(0, len(brain_datas)):
        brain_datas[run_index].X = dsg_mats[run_index]
    
    # calculate Beta Values per run
    stats = []
    for run_index in range(0, len(brain_datas)):
        stats.append(brain_datas[run_index].regress())
        
    return stats, dsg_mats

def change_event_typeForLSA(events_):    
    """
    convert event_type for doing lsa
    
    :param events: list of event
    
    return dataframe
    """
    # events
    events = copy.deepcopy(events_)
    
    # info data
    info_datas = []
    for run_index in range(0, len(events)):
        info_data = {}
        for stimulus_type in np.unique(events[run_index]["trial_type"]):
            info_data[stimulus_type] = 0            
        info_datas.append(info_data)
    
    # Iterate all runs
    for run_index in range(0, len(events)):
        stimulus_column = events[run_index]["trial_type"]
        info_data = info_datas[run_index]

        temp_stimulus_column = [] # Saving event type converted from all event
        
        # Iterate all stimulus
        for stimulus_i in range(0, len(stimulus_column)):
            stimulus = stimulus_column[stimulus_i]
            
            if stimulus == "+":
                temp_stimulus_column.append(stimulus + "_" + str(info_data[stimulus]))
                info_data[stimulus] = info_data[stimulus] + 1
            else:
                # Convert event stimulus condition type
                temp_stimulus_column.append(stimulus + "_" + str(info_data[stimulus]))
                info_data[stimulus] = info_data[stimulus] + 1
        
        # Assign converted event type
        events[run_index]["trial_type"] = temp_stimulus_column
            
    return events

def change_event_typeForLSS(lsa_event, target_condition, trial_index):
    """
    Change event type to do LSS
    
    :param lsa_event: event(dataframe)
    :param target_condition: condition
    :param trial_index: index
    
    return event
    """
    event = lsa_event.copy()
    target_conditionWithTrial = sj_file_system.str_join([target_condition, str(trial_index)])

    return sj_preprocessing.change_df(event, "trial_type", lambda t_type: "Nuisance" if t_type != target_conditionWithTrial else target_conditionWithTrial)

def compare_design_mats(design_mats1, design_mats2, mat1_description="", mat2_description=""):
    """
    Compare design matrices between design_mats1 and design_mats2 to use drawing matrix.
    
    :param design_mats1: design matricies(list - df)
    :param design_mats2: design matricies(list - df)
    :param mat1_description: (string)
    :param mat1_description: (string)
    """
    assert len(design_mats1) == len(design_mats2), "Please match list size"
    
    run_length = len(design_mats1)
    
    fig, axes = plt.subplots(nrows=2, ncols=run_length) # nrow=2: (origin, parametric modulation)
    fig.set_size_inches(30, 30)

    dm1_axis_index = 0
    dm2_axis_index = 1
    
    for run_index in range(run_length):
        dm1_axis = axes[dm1_axis_index][run_index]
        plot_design_matrix(design_mats1[run_index], ax = dm1_axis)
        dm1_axis.set_title(sj_file_system.str_join([mat1_description, str(run_index + 1)]))

    for run_index in range(run_length):
        dm2_axis = axes[dm2_axis_index][run_index]
        plot_design_matrix(design_mats2[run_index], ax = dm2_axis)
        dm2_axis.set_title(sj_file_system.str_join([mat2_description, str(run_index + 1)]))

def compare_design_mats_hemodynamic(design_mats1, 
                        design_mats2, 
                        conditions,
                        mat1_description = "design1", 
                        mat2_description = "design2",
                        ylim = (-0.5, 5)):
    """
    Compare design matrices between design_mats1 and design_mats2 to use drawing hemodynamic response.
    
    :param design_mats1: design matricies(list - df)
    :param design_mats2: design matricies(list - df)
    :param conditions: conditions(list)
    :param mat1_description: (string)
    :param mat1_description: (string)
    :param ylim: limination of y-axis(tuple)
    """
    assert len(design_mats1) == len(design_mats2), "Please match list size"
    
    run_length = len(design_mats1)
    
    fig, axes = plt.subplots(nrows=run_length, ncols=4)
    fig.set_size_inches(30, 20)

    for run_index in range(run_length):
        axis_index = 0
        """
        Design Matrix1
        """
        dsg_mat1 = design_mats1[run_index]
        move_conditions_data = np.array(list(map(lambda condition: dsg_mat1[str(condition)].to_numpy(), conditions))).T
        move_legends = list(map(lambda condition: str(condition), conditions))
        plot_timeseries(axis = axes[run_index][axis_index], 
                        data = move_conditions_data, 
                        labels = move_legends,
                        title= sj_file_system.str_join([mat1_description, "run", str(run_index + 1), "move"]),
                        ylim=ylim)
        axis_index += 1

        rest_conditions_data = np.array(list(map(lambda condition: dsg_mat1["+_" + str(condition)].to_numpy(), conditions))).T
        rest_legends = list(map(lambda condition: "+_" + str(condition), conditions))
        plot_timeseries(axis = axes[run_index][axis_index], 
                        data = rest_conditions_data, 
                        labels = rest_legends, 
                        title= sj_file_system.str_join([mat1_description, "run", str(run_index + 1), "rest"]),
                        ylim=ylim)
        axis_index += 1

        """
        Design Matrix2
        """
        dsg_mat2 = design_mats2[run_index]
        move_conditions_data = np.array(list(map(lambda condition: dsg_mat2[str(condition)].to_numpy(), conditions))).T
        move_legends = list(map(lambda condition: str(condition), conditions))
        plot_timeseries(axis = axes[run_index][axis_index], 
                        data = move_conditions_data, 
                        labels = move_legends,
                        title= sj_file_system.str_join([mat2_description, "run", str(run_index + 1), "move"]),
                        ylim=ylim)
        axis_index += 1

        rest_conditions_data = np.array(list(map(lambda condition: dsg_mat2["+_" + str(condition)].to_numpy(), conditions))).T
        rest_legends = list(map(lambda condition: "+_" + str(condition), conditions))
        plot_timeseries(axis = axes[run_index][axis_index], 
                        data = rest_conditions_data, 
                        labels = rest_legends, 
                        title= sj_file_system.str_join([mat2_description, "run", str(run_index + 1), "rest"]),
                        ylim=ylim)
        axis_index += 1

    plt.tight_layout()

def searchlight_with_beta(Xs, 
                          Ys, 
                          full_mask, 
                          subj_name, 
                          searchlight_dir_path, 
                          n_jobs = 1, 
                          radius=6, 
                          estimator = "svc",
                          prefix = ""):
    """
    Do searchlight Decoding analysis using beta values
    
    :param Xs: list of nifti image(list) seperated by run, shape (n_x, n_y, n_z, n_conditions)
    :param Ys: list of label(list) seperated by run ex) [ [condition1, condition1, condition2, condition2], [condition1, condition1, condition2, condition2] ]
    :param full_mask: full_mask(nifti image)
    :param searchlight_dir_path: directory path where the result is located.
    :param n_jobs: n_jobs
    :param radius: radius
    :param estimator: ‘svr’, ‘svc’, or an estimator object implementing ‘fit’
    :param prefix: save file prefix(string)
    
    return searchlight_obj
    """
    
    start = datetime.datetime.now()
    print(start)

    cv = GroupKFold(len(Xs))
    
    groups = []
    for run_i in range(0, len(Xs)):
        for _ in range(0, Xs[run_i].shape[-1]):
            groups.append(run_i)
    groups = np.array(groups)

    Xs = image.concat_imgs(Xs)
    Ys = np.concatenate(Ys)
    
    preproc_np_data = np.nan_to_num(zscore(Xs.get_fdata(), axis=-1)) 
    Xs = nb.Nifti1Image(preproc_np_data, 
                        full_mask.affine, 
                        full_mask.header)
    
    if estimator == "LDA":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        estimator_clf = LinearDiscriminantAnalysis()
    else:
        estimator_clf = estimator
        
    # Make Model
    searchlight = nilearn.decoding.SearchLight(
        full_mask,
        radius=radius, 
        n_jobs=n_jobs,
        verbose=False,
        cv=cv,
        estimator = estimator_clf,
        scoring="balanced_accuracy")

    # Fitting Model
    searchlight.fit(imgs=Xs, y=Ys, groups=groups)
    
    end = datetime.datetime.now()
    
    # Save
    save_file_name = sj_file_system.str_join([prefix, subj_name, estimator, "searchlight_clf"], deliminator = "_")   
    sj_file_system.save(searchlight, save_file_name)
    
    score_img = image.new_img_like(ref_niimg = full_mask, data = searchlight.scores_)
    nb.save(score_img, os.path.join(searchlight_dir_path, save_file_name + ".nii"))
    
    print(start, end)
    
    return searchlight

def apply_func_rdmBrain(rdm_brain, func):
    """
    Apply function to rdm brain
    
    :param rdm_brain: rdm_brain(list) - shape (nx, ny, nz, n_cond, n_cond)
    :param func: function to apply rdm
    
    return list(matched with brain shape)
    """
    
    nx, ny, nz = rdm_brain.shape[0], rdm_brain.shape[1], rdm_brain.shape[2]

    result = np.zeros([nx,ny,nz]).tolist()
    for i in range(nx):
        for j in range(ny):
            for z in range(nz):
                result[i][j][z] = func(rdm_brain[i][j][z])
    return result

def image2referenceCoord(ijk, affine):
    """
    change image coordinate to reference coordinate
    reference coordinate can be scanner coordinate or MNI coordinate...
    
    :param ijk: image coordinate(np.array)
    :param affine: affine matrix(np.array)
    
    return scanner coordinate(np.array)
    """
    return np.matmul(affine, np.array([ijk[0], ijk[1], ijk[2], 1]))[0:3]

def reference2imageCoord(xyz, affine):
    """
    change reference coordinate to image coordinate
    reference coordinate can be scanner coordinate or MNI coordinate...
    
    :param xyz: anatomical coordinate(np.array)
    :param affine: affine matrix(np.array)
    
    return image coordinate(np.array)
    """
    
    result = np.matmul(np.linalg.inv(affine), [xyz[0], xyz[1], xyz[2], 1])[0:3]
    result = np.ceil(result).astype(int) # note: This is ad-hoc process - (np.ceil)
    return result

def LPSp_toRASp(xyz):
    """
    Convert LPS+ coordinate to RAS+ coordinate
    
    :param xyz: LPS+ coord(list)
    
    return xyz(list)
    """
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    
    return -x, -y, z

def RASp_toLPSp(xyz):
    """
    Convert RAS+ coordinate to LPS+ coordinate
    
    :param xyz: RAS+ coord(list)
    
    return xyz(list)
    """
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    
    return -x, -y, z

def image3d_to_1d(ijk, shape_3d):
    """
    Convert 3d coord to 1d coord
    
    :param ijk: index of image(list or array) ex) [1,2,3]
    :param shape: shape of 3d fmri image(list) ex) [96, 114, 96]
    
    return 1d coord
    """
    i = ijk[0]
    j = ijk[1]
    k = ijk[2]
    
    ni = shape_3d[0]
    nj = shape_3d[1]
    nk = shape_3d[2]
    
    return i * (nj * nk) + j * nk + k

def image1d_to_3d(index, shape_3d):
    """
    Convert 1d coord to 3d coord
    
    :param index: index of 1d image ex) [3]
    :param shape: shape of 3d fmri image(list) ex) [96, 114, 96]
    
    return 3d coord
    """
    ni = shape_3d[0]
    nj = shape_3d[1]
    nk = shape_3d[2]
    
    i = int(index / (nj * nk))
    j = int((index - i * nj * nk) / nk)
    k = (index - i * nj * nk) - (j * nk)
    
    return i, j, k

def get_uniquePattern(conds):
    """
    Get unique pattern from conditions
    
    :param conds: conditions(list)
    
    return unique pattern of conditions(keep ordering)
    """
    
    return list(dict.fromkeys(conds))

def sort_rdm(rdm_array, origin_conditions, reordered_conditions):
    """
    Sort rdm by reordered_conditions
    
    :param rdm_array: rdm array(2d array)
    :param origin_conditions: list of condition(1d list)
    :param reordered_conditions: list of condition(1d list)
    
    retrun sorted rdm(2d array)
    """
    cond_length = len(origin_conditions)
    
    pattern_info = dict(zip(origin_conditions, np.arange(0, cond_length)))
    
    re_order_indexes = [pattern_info[cond] for cond in reordered_conditions]
    
    origin_axis1, origin_axis2 = np.meshgrid(origin_conditions, origin_conditions)
    convert_axis1, convert_axis2 = np.meshgrid(reordered_conditions, reordered_conditions)
    
    orgin_grid = np.zeros((cond_length, cond_length)).tolist()
    sorted_grid = np.zeros((cond_length, cond_length)).tolist()
    result_grid = np.zeros((cond_length, cond_length)).tolist()
    
    for i in range(cond_length):
        for j in range(cond_length):
            orgin_grid[i][j] = (origin_axis2[i][j], origin_axis1[i][j])
            sorted_grid[i][j] = (convert_axis2[i][j], convert_axis1[i][j])

    for i in range(cond_length):
        for j in range(cond_length):
            target_pair = sorted_grid[i][j]
            target_array = np.array(sj_higher_function.recursive_mapWithDepth(orgin_grid, 
                                                                              lambda x: x == target_pair, 
                                                                              1))

            x_indexes, y_indexes = np.where(target_array == True)
            assert len(x_indexes) == 1 and len(y_indexes) == 1, "Please check duplicate"
            x_i = x_indexes[0]
            y_i = y_indexes[0]

            result_grid[i][j] = rdm_array[x_i][y_i]
            
    return np.array(result_grid)

if __name__ == "__main__":

    result = sj_brain.highlight_stat(roi_array=motor_left_mask.get_data(),
                                     stat_array=np.load(
                                         "/Users/yoonseojin/statistics_sj2/CLMN/Replay_Exp/experiment/20210407_blueprint_0324v2/HR01/searchlight/preprocessed_2mm/HR01_searchlight_interest_10.npy"),
                                     stat_threshold=0.6)
    plotting.view_img(nb.Nifti1Image(result["data"], full_mask.affine, full_mask.header),
                      anat,
                      cmap=result["color_map"])

    result = sj_brain.colored_roi_with_stat(
        roi_arrays=[mask_left_precentral_gyrus.get_fdata(), mask_occipital_cortex.get_fdata(),
                    mask_all_hippocampus.get_fdata()],
        stat_map=np.load(
            "/Users/yoonseojin/statistics_sj2/CLMN/Replay_Exp/experiment/20210407_blueprint_0324v2/HR01/searchlight/preprocessed_2mm/HR01_searchlight_interest_2.npy"),
        stat_threshold=0.60)

    plotting.view_img(nb.Nifti1Image(result["data"], full_mask.affine, full_mask.header),
                      anat,
                      cmap=result["color_map"])

    upper_tri(np.repeat(3, 9).reshape(3,3))
    
    a = RDM_model(np.array([1,0,1,0]).reshape(2,2), "transition", ["1","2"])
    RDM_brains = RSA(models=[a],
                     conditions=["!", "@", "#"],
                     full_mask=full_mask,
                     datas = beta_values,
                     save_rdm_path = os.path.join(output_dir_path, "rdm"),
                     save_corr_brain_path=os.path.join(output_dir_path, "corr_brain"),
                     n_jobs=3
                    )
    
    make_RDM_brain(brain_shape, rdm)
    
    reference_img = image.mean_img(fMRI_datas[0])

    mask_manager = fan_roi_mask_manager(fan_info_path=mask_fan_info_path, 
                                               mask_dir_path=mask_dir_path, 
                                               reference_img=image.mean_img(fMRI_datas[0]))
    
    mask_manager.search_roi(["precentral gyrus", "Lt"])
    mask_manager.search_mask_info(["precentral gyrus", "Lt"])
    
    parcellation_roi_manager = parcellation_roi_mask_manager(parcellation_info_path = parcellation_info_path, 
                                  mask_path = os.path.join(mask_dir_path, "parcellation", "Neurosynth_parcellation_k50_2mm.nii.gz"), 
                                  reference_img= full_mask)
    
    image2referenceCoord([0,0,0], full_mask.affine)
    reference2imageCoord([0,0,0], full_mask.affine)
    
    rdm = np.array(
        [
            [0,2,3],
            [4,0,6],
            [7,8,0],
        ]
    )
    sort_rdm(rdm, ["A", "B", "C"], ["B", "A", "C"])