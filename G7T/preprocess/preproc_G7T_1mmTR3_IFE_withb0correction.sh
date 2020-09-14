#!/bin/bash
export ANTSPATH=/Users/clmnlab/install/bin
export PATH=${PATH}:/Users/clmnlab/install/bin
subjlist=( G7T_IFE18 )
group=G7T_IF

for subj in ${subjlist[@]}
do
    outputdir=/Volumes/clmnlab/G7T/main/fmri_data/preproc_data/7T_anat/$subj
    # outputdir=~/Documents/fmri/practice/output/7T_prac/$subj
    anat1_dir=~/Documents/fmri/practice/raw/$group/$subj/MP2RAGE*INV1*
    echo $anat1_dir
    anat2_dir=~/Documents/fmri/practice/raw/$group/$subj/MP2RAGE*INV2*
    anat3_dir=~/Documents/fmri/practice/raw/$group/$subj/MP2RAGE*UNI*
    atlas_dir=~/Documents/fmri/practice/template/*Atlas*
    run1_dir=~/Documents/fmri/practice/raw/$group/$subj/GE1ISO*RUN1*
    run2_dir=~/Documents/fmri/practice/raw/$group/$subj/GE1ISO*RUN2*
    run3_dir=~/Documents/fmri/practice/raw/$group/$subj/GE1ISO*RUN3*
    run4_dir=~/Documents/fmri/practice/raw/$group/$subj/GE1ISO*RUN4*
    run5_dir=~/Documents/fmri/practice/raw/$group/$subj/GE1ISO*RUN5*
    run6_dir=~/Documents/fmri/practice/raw/$group/$subj/GE1ISO*RUN6*
    PA_dir=~/Documents/fmri/practice/raw/$group/$subj/GE1ISO_3X3_PA_180*
    AP_dir=~/Documents/fmri/practice/raw/$group/$subj/GE1ISO_3X3_AP_0*
    rest_dir=~/Documents/fmri/practice/raw/$group/$subj/GE1ISO*RESTING*

    res=1
    fwhm=2  # 4
    thresh_motion=0.5
    runs=`count -digits 2 1 6`

    if (! -d $outputdir ) 
    then
    mkdir $outputdir
    else
    echo "output dir ${outputdir} already exists"
    fi

    cd $rest_dir
    Dimon -infile_pat '*.IMA' -gert_create_dataset -gert_to3d_prefix temp \
    -gert_outdir $outputdir -gert_quit_on_err
    3dWarp -deoblique -prefix $outputdir/func.$subj.resting.r00 $outputdir/temp+orig
    rm $outputdir/temp*

    cd $run1_dir
    Dimon -infile_pat '*.IMA' -gert_create_dataset -gert_to3d_prefix temp \
    -gert_outdir $outputdir -gert_quit_on_err
    3dWarp -deoblique -prefix $outputdir/func.$subj.run01.r00 $outputdir/temp+orig
    rm $outputdir/temp*

    cd $run2_dir
    Dimon -infile_pat '*.IMA' -gert_create_dataset -gert_to3d_prefix temp \
    -gert_outdir $outputdir -gert_quit_on_err
    3dWarp -deoblique -prefix $outputdir/func.$subj.run02.r00 $outputdir/temp+orig
    rm $outputdir/temp*

    cd $run3_dir
    Dimon -infile_pat '*.IMA' -gert_create_dataset -gert_to3d_prefix temp \
    -gert_outdir $outputdir -gert_quit_on_err
    3dWarp -deoblique -prefix $outputdir/func.$subj.run03.r00 $outputdir/temp+orig
    rm $outputdir/temp*

    cd $run4_dir
    Dimon -infile_pat '*.IMA' -gert_create_dataset -gert_to3d_prefix temp \
    -gert_outdir $outputdir -gert_quit_on_err
    3dWarp -deoblique -prefix $outputdir/func.$subj.run04.r00 $outputdir/temp+orig
    rm $outputdir/temp*

    cd $run5_dir
    Dimon -infile_pat '*.IMA' -gert_create_dataset -gert_to3d_prefix temp \
    -gert_outdir $outputdir -gert_quit_on_err
    3dWarp -deoblique -prefix $outputdir/func.$subj.run05.r00 $outputdir/temp+orig
    rm $outputdir/temp*

    cd $run6_dir
    Dimon -infile_pat '*.IMA' -gert_create_dataset -gert_to3d_prefix temp \
    -gert_outdir $outputdir -gert_quit_on_err
    3dWarp -deoblique -prefix $outputdir/func.$subj.run06.r00 $outputdir/temp+orig
    rm $outputdir/temp*

    cd $PA_dir
    Dimon -infile_pat '*.IMA' -gert_create_dataset -gert_to3d_prefix temp \
    -gert_outdir $outputdir -gert_quit_on_err
    3dWarp -deoblique -prefix $outputdir/func.$subj.PA.r00 $outputdir/temp+orig
    rm $outputdir/temp*

    cd $AP_dir
    Dimon -infile_pat '*.IMA' -gert_create_dataset -gert_to3d_prefix temp \
    -gert_outdir $outputdir -gert_quit_on_err
    3dWarp -deoblique -prefix $outputdir/func.$subj.AP.r00 $outputdir/temp+orig
    rm $outputdir/temp*

    cd $anat1_dir
    Dimon -infile_pat '*.IMA' -gert_create_dataset -gert_to3d_prefix temp \
    -gert_outdir $outputdir -gert_quit_on_err
    3dWarp -deoblique -prefix $outputdir/MP2RAGE.$subj.INV1.r00 $outputdir/temp+orig
    rm $outputdir/temp*

    cd $anat2_dir
    Dimon -infile_pat '*.IMA' -gert_create_dataset -gert_to3d_prefix temp \
    -gert_outdir $outputdir -gert_quit_on_err
    3dWarp -deoblique -prefix $outputdir/MP2RAGE.$subj.INV2.r00 $outputdir/temp+orig
    rm $outputdir/temp*

    cd $anat3_dir
    Dimon -infile_pat '*.IMA' -gert_create_dataset -gert_to3d_prefix temp \
    -gert_outdir $outputdir -gert_quit_on_err
    3dWarp -deoblique -prefix $outputdir/MP2RAGE.$subj.UNI.r00 $outputdir/temp+orig
    rm $outputdir/temp*

    cd $outputdir
    for run in $runs
    do
        3dAFNItoNIFTI func.$subj.run$run.r00+orig
    done
    3dAFNItoNIFTI MP2RAGE.$subj.INV1.r00+orig
    3dAFNItoNIFTI MP2RAGE.$subj.INV2.r00+orig
    3dAFNItoNIFTI MP2RAGE.$subj.UNI.r00+orig

    fslmaths MP2RAGE.$subj.INV1*nii -mul MP2RAGE.$subj.INV2*nii MP2RAGE.$subj.UNIxINV2 -odt float

    antsBrainExtraction.sh -d 3 -a MP2RAGE.$subj.UNIxINV2.nii.gz \
    -e $atlas_dir/T_template0.nii.gz \
    -m $atlas_dir/T_template0_BrainCerebellumProbabilityMask.nii.gz \
    -f $atlas_dir/T_template0_BrainCerebellumRegistrationMask.nii.gz \
    -o MP2RAGE_

    # register epi with each other
    for run in $runs
    do
        3dTcat -prefix $outputdir/pb00.$subj.run$run.tcat $outputdir/func.$subj.run$run.r00+orig
    done

    touch out.pre_ss_warn.txt
    npol=4

    for run in $runs
    do
        3dToutcount -automask -fraction -polort $npol -legendre  \
        $outputdir/pb00.$subj.run$run.tcat+orig > $outputdir/outcount.$subj.run$run.1D
        if ( `1deval -a $outputdir/outcount.$subj.run01.1D"{0}" -expr "step(a-0.4)"` ) 
        then
        echo "** TR #0 outliers: possible pre-steady state TRs in run01" \
        >> out.pre_ss_warn.txt
        fi
    done

    cat outcount.$subj.run*.1D > outcount_rall.$subj.1D

    for run in $runs
    do
        3dDespike -NEW -nomask -prefix $outputdir/pb00.$subj.run$run.despike $outputdir/pb00.$subj.run$run.tcat+orig
        3dTshift -tzero 0 -quintic -prefix $outputdir/pb01.$subj.run$run.tshift $outputdir/pb00.$subj.run$run.despike+orig
    done

    3dcopy MP2RAGE_BrainExtractionBrain.nii.gz $subj.sSanat

    # 3dUnifize
    3dUnifize -input $subj.sSanat+orig -prefix $subj.UnisSanat -GM
    3dAFNItoNIFTI $subj.UnisSanat+orig

    ################################################################################
    ################################################################################


    3dTcat -prefix $outputdir/pb00.$subj.PA.tcat $outputdir/func.$subj.PA.r00+orig
    3dAFNItoNIFTI $outputdir/pb00.$subj.PA.tcat+orig
    gzip -1v $outputdir/pb00.$subj.PA.tcat.nii

    3dTcat -prefix $outputdir/pb00.$subj.AP.tcat $outputdir/func.$subj.AP.r00+orig
    3dAFNItoNIFTI $outputdir/pb00.$subj.AP.tcat+orig
    gzip -1v $outputdir/pb00.$subj.AP.tcat.nii

    # copy external -blip_forward_dset dataset
    3dTcat -prefix $outputdir/blip_forward $outputdir/pb00.$subj.AP.tcat+orig
    # copy external -blip_reverse_dset dataset
    3dTcat -prefix $outputdir/blip_reverse $outputdir/pb00.$subj.PA.tcat+orig


    # ================================== blip ==================================
    # compute blip up/down non-linear distortion correction for EPI

    # create median datasets from forward and reverse time series
    3dTstat -median -prefix rm.blip.med.fwd blip_forward+orig
    3dTstat -median -prefix rm.blip.med.rev blip_reverse+orig

    # automask the median datasets
    3dAutomask -apply_prefix rm.blip.med.masked.fwd rm.blip.med.fwd+orig
    3dAutomask -apply_prefix rm.blip.med.masked.rev rm.blip.med.rev+orig

    # compute the midpoint warp between the median datasets
    3dQwarp -plusminus -pmNAMES Rev For                           \
    -pblur 0.05 0.05 -blur -1 -1                          \
    -noweight -minpatch 9                                 \
    -source rm.blip.med.masked.rev+orig                   \
    -base   rm.blip.med.masked.fwd+orig                   \
    -prefix blip_warp

    # warp median datasets (forward and each masked) for QC checks
    3dNwarpApply -quintic -nwarp blip_warp_For_WARP+orig          \
    -source rm.blip.med.fwd+orig                     \
    -prefix blip_med_for

    3dNwarpApply -quintic -nwarp blip_warp_For_WARP+orig          \
    -source rm.blip.med.masked.fwd+orig              \
    -prefix blip_med_for_masked

    3dNwarpApply -quintic -nwarp blip_warp_Rev_WARP+orig          \
    -source rm.blip.med.masked.rev+orig              \
    -prefix blip_med_rev_masked

    # warp EPI time series data
    for run in $runs
    do
        3dNwarpApply -quintic -nwarp blip_warp_For_WARP+orig      \
        -source pb01.$subj.run$run.tshift+orig         \
        -prefix pb01.$subj.run$run.blip
    done

    ######################################################################################################
    ######################################################################################################


    # =============================== align ==================================
    # for e2a: compute anat alignment transformation to EPI registration base
    # (new anat will be intermediate, stripped, epi_$subjID.anat_ns+orig)
    align_epi_anat.py -anat2epi -anat $subj.UnisSanat+orig -anat_has_skull no                   \
        -epi pb01.$subj.run01.tshift+orig -epi_base 2 -epi_strip  3dAutomask         \
        -suffix _al_junk -check_flip -volreg off -tshift off -ginormous_move       \
        -deoblique off                                                             \
        -cost nmi  -align_centers yes
    # -cost nmi : weired result in the multiband8 protocol
    # -cost lpa (local pearson correlation)



    # ================================== tlrc ==================================
    # warp anatomy to standard space
    @auto_tlrc -base MNI152_T1_2009c+tlrc -input $subj.UnisSanat+orig -no_ss
    # store forward transformation matrix in a text file
    cat_matvec $subj.UnisSanat+tlrc::WARP_DATA -I > warp.anat.Xat.1D


    # ================================= volreg =================================
    # align each dset to base volume, align to anat, warp to tlrc space

    # verify that we have a +tlrc warp dataset
    if [ -f ${subj}.UnisSanat+tlrc.HEAD ] ; then
        echo 'exist'
    else
        echo "** missing +tlrc warp dataset: $subj.UnisSanat+tlrc.HEAD"
        exit
    fi

    #================================== register and warp =======================================
    for run in $runs
    do
        # register each volume to the base
        3dvolreg -verbose -zpad 1 -cubic -base pb01.$subj.run01.tshift+orig'[2]' 			 \
        -1Dfile dfile.run$run.1D -prefix rm.epi.volreg.run$run \
        -1Dmatrix_save mat.run$run.vr.aff12.1D  \
        pb01.$subj.run$run.blip+orig

        # create an all-1 dataset to mask the extents of the warp
        3dcalc -overwrite -a pb01.$subj.run$run.blip+orig -expr 1 -prefix rm.epi.all1
        


        # catenate volreg, epi2anat and tlrc transformations
        cat_matvec -ONELINE $subj.UnisSanat+tlrc::WARP_DATA -I $subj.UnisSanat_al_junk_mat.aff12.1D -I  \
        mat.run$run.vr.aff12.1D > mat.run$run.warp.aff12.1D
        
        # apply catenated xform : volreg, epi2anat and tlrc
        3dAllineate -base $subj.UnisSanat+tlrc \
        -input pb01.$subj.run$run.blip+orig \
        -1Dmatrix_apply mat.run$run.warp.aff12.1D \
        -mast_dxyz $res -prefix rm.epi.nomask.run$run
        
        # warp the all-1 dataset for extents masking
        3dAllineate -base $subj.UnisSanat+tlrc \
        -input rm.epi.all1+orig \
        -1Dmatrix_apply mat.run$run.warp.aff12.1D     \
        -mast_dxyz $res -final NN -quiet \
        -prefix rm.epi.1.run$run
        
        # make an extents intersection mask of this run
        3dTstat -min -prefix rm.epi.min.run$run rm.epi.1.run$run+tlrc
        # 4d(epi.1) -> 3d(epi.min)
    done

    # make a single file of registration params
    cat dfile.run*.1D > dfile_rall.1D # YJS comment: concatenating motion parameters of all runs

    # create the extents mask: mask_epi_extents+tlrc
    # (this is a mask of voxels that have valid data at every TR)
    # (only 1 run, so just use 3dcopy to keep naming straight)
    3dcopy rm.epi.min.run01+tlrc mask_epi_extents

    # and apply the extents mask to the EPI data
    # (delete any time series with missing data)
    for run in $runs
    do
        3dcalc -a rm.epi.nomask.run$run+tlrc -b mask_epi_extents+tlrc -expr 'a*b' -prefix pb02.$subj.run$run.volreg
    done

    # create an anat_final dataset, aligned with stats
    3dcopy $subj.UnisSanat+tlrc anat_final.$subj


    # -----------------------------------------
    # warp anat follower datasets (affine)
    3dAllineate -source $subj.UnisSanat+orig \
    -master anat_final.$subj+tlrc \
    -final wsinc5 -1Dmatrix_apply warp.anat.Xat.1D \
    -prefix anat_w_skull_warped

    # ================================================= blur =================================================
    # blur each volume of each run
    for run in $runs
    do
        3dmerge -1blur_fwhm $fwhm -doall -prefix pb03.$subj.run$run.blur pb02.$subj.run$run.volreg+tlrc
    done

    # ================================================= mask =================================================
    # create 'full_mask' dataset (union mask)
    for run in $runs
    do
        3dAutomask -dilate 1 -clfrac 0.3 -prefix rm.mask_run$run pb03.$subj.run$run.blur+tlrc
    done

    # create union of inputs, output type is byte
    3dmask_tool -inputs rm.mask_r*+tlrc.HEAD -union -prefix full_mask.$subj
    # ---- create subject anatomy mask, mask_anat.$subj+tlrc ----
    #      (resampled from tlrc anat)
    3dresample -master full_mask.$subj+tlrc -input $subj.UnisSanat+tlrc -prefix rm.resam.anat
    # convert to binary anat mask; fill gaps and holes
    3dmask_tool -dilate_input 5 -5 -fill_holes -input rm.resam.anat+tlrc -prefix mask_anat.$subj

    # ================================= scale ==================================
    # scale each voxel time series to have a mean of 100 (be sure no negatives creep in)
    # (subject to a range of [0,200])
    for run in $runs
    do
        3dTstat -prefix rm.mean_run$run pb03.$subj.run$run.blur+tlrc
        3dcalc -float -a pb03.$subj.run$run.blur+tlrc -b rm.mean_run$run+tlrc -c mask_epi_extents+tlrc  \
            -expr 'c * min(200, a/b*100)*step(a)*step(b)' -prefix pb04.$subj.run$run.scale
    done

    # ================================ regress =================================
    # compute de-meaned motion parameters (for use in regression)
    1d_tool.py -infile dfile_rall.1D -set_nruns 1 -demean -write motion_demean.$subj.1D
    # compute motion parameter derivatives (just to have)
    1d_tool.py -infile dfile_rall.1D -set_nruns 1 -derivative -demean -write motion_deriv.$subj.1D
    # create censor file motion_${subj}_censor.1D, for censoring motion
    1d_tool.py -infile dfile_rall.1D -set_nruns 1 -show_censor_count -censor_prev_TR -censor_motion $thresh_motion motion_$subj
    for run in $runs
    do
        1d_tool.py -infile dfile.run$run.1D -set_nruns 1 -demean -write motion_demean.$subj.run$run.1D
        1d_tool.py -infile dfile.run$run.1D -set_nruns 1 -derivative -demean -write motion_deriv.$subj.run$run.1D
        1d_tool.py -infile dfile.run$run.1D -set_nruns 1 -show_censor_count -censor_prev_TR -censor_motion $thresh_motion motion_$subj.run$run
    done

    # ================================ bandpass filtering  =================================
    for run in $runs
    do
    #	3dTproject -polort 0 -input pb04.$subj.r$run.scale+tlrc.HEAD -mask full_mask.$subj+tlrc -passband 0.01 0.1 \
    #	-censor motion_${subj}.r{$run}_censor.1D -cenmode ZERO -ort motion_deriv.$subj.r$run.1D  -prefix bp.$subj.r$run
        3dTproject -polort 0 -input pb04.$subj.run$run.scale+tlrc.HEAD -mask full_mask.$subj+tlrc -passband 0.01 0.1 \
        -censor motion_$subj.run${run}_censor.1D -cenmode ZERO -ort motion_demean.$subj.run$run.1D  -prefix bp_demean.$subj.run$run
    done
    # ================== auto block: generate review scripts ===================
    # generate a review script for the unprocessed EPI data
    gen_epi_review.py -script @epi_review.$subj -dsets pb00.$subj.run*.tcat+orig.HEAD

done

