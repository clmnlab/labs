for f in $(ls *.nii.gz); do echo $f $(fslinfo $f | grep "^dim4"); done
