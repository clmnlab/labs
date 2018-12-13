"""
This code is based on nilearn's searchlight code.
- https://github.com/nilearn/nilearn/blob/master/nilearn/decoding/searchlight.py

written by eyshin05 (eyshin05@gmail.com)
"""

import numpy as np

from sklearn.base import BaseEstimator

from nilearn import masking
from nilearn.image.resampling import coord_transform
from nilearn.input_data.nifti_spheres_masker import _apply_mask_and_get_affinity
from sklearn.model_selection import cross_val_score


class SpatioTemporalSearchLight(BaseEstimator):
    """Implemented spatiotemporal searchlight analysis using an arbitrary type of classifier
    This class doesn't support parallel jobs.

    Parameters
    -----------
    mask_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        boolean image giving location of voxels containing usable signals.

    radius : float, optional
        radius of the searchlight ball, in millimeters. Defaults to 2.

    estimator : an estimator object implementing 'fit'
        The object to use to fit the data

    cv : A cross-validation generator.
    """

    def __init__(self, mask_img, estimator, cv, radius=2., scoring=None):
        self.mask_img = mask_img
        self.radius = radius
        self.scoring = scoring

        if estimator is None:
            raise ValueError('This instance need an estimator instance')
        else:
            self.estimator = estimator

        if cv is None:
            raise ValueError('This instance need an cv instance')
        else:
            self.cv = cv

    def _search_light(self, X, y, A, scoring=None, groups=None):
        """Function for computing a search_light

        Parameters
        ----------
        X : array-like of shape 3D data [trial, masked_voxels, time series] to fit.

        y : array-like
            target variable to predict.

        A : scipy sparse matrix.
            adjacency matrix. Defines for each feature the neigbhoring features
            following a given structure of the data.

        cv : cross-validation generator.

        groups : array-like, optional
            group label for each sample for cross validation. default None
            NOTE: will have no effect for scikit learn < 0.18

        Returns
        -------
        scores : array-like of shape (number of rows in A) search_light scores
        """

        list_rows = A.rows
        X = np.swapaxes(X, 0, 1)  # X dims: (trial, temporal, spatial)
        trial_size = X.shape[0]

        par_scores = np.zeros(len(list_rows))

        for i, row in enumerate(list_rows):
            kwargs = dict()
            kwargs['scoring'] = scoring
            kwargs['groups'] = groups

            par_scores[i] = np.mean(cross_val_score(self.estimator,
                                                    X[:, :, row].reshape(trial_size, -1),
                                                    y, cv=self.cv, n_jobs=1, **kwargs))

        return par_scores

    def fit(self, imgs, y, groups=None):
        """Fit the spatiotemporal searchlight

        Parameters
        ----------
        imgs : Niimg-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
            Use 5D image. [x voxels, y voxels, z voxels, trials, time series of each trials]

        y : 1D array-like
            Target variable to predict. Must have exactly as many elements as trials in imgs.

        groups : array-like, optional
            group label for each sample for cross validation. Must have
            exactly as many elements as trials in imgs. default None
            NOTE: will have no effect for scikit learn < 0.18 (as nilearn says)
        """

        # check if image is 5D, simply.
        if len(imgs.shape) != 5:
            raise ValueError('This SpatioTemporalSearhcLight instance needs 5D image.')

        # Get the seeds
        process_mask_img = self.mask_img

        # Compute world coordinates of the seeds
        process_mask, process_mask_affine = masking._load_mask_img(process_mask_img)
        process_mask_coords = np.where(process_mask != 0)
        process_mask_coords = coord_transform(
            process_mask_coords[0], process_mask_coords[1],
            process_mask_coords[2], process_mask_affine)
        process_mask_coords = np.asarray(process_mask_coords).T

        X, A = _apply_mask_and_get_affinity(
            process_mask_coords, imgs, self.radius, True,
            mask_img=self.mask_img)

        # Run Searchlight
        scores = self._search_light(X, y, A, groups=groups)
        scores_3D = np.zeros(process_mask.shape)
        scores_3D[process_mask] = scores

        self.scores_ = scores_3D

        return self