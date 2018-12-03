#!/bin/bash
# Size of 4th dimention in the fMRI data (raw images, beta values, or etc...)
for f in $(ls *.nii.gz); do echo $f $(fslinfo $f | grep "^dim4"); done
