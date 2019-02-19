import clmnlab_libs.mvpa_toolkits as mtk
import glob
import nilearn.image


if __name__ == '__main__':
    subj_list = [
        'IN04', 'IN05', 'IN07', 'IN09', 'IN10', 'IN11',
        'IN12', 'IN13', 'IN14', 'IN15', 'IN16', 'IN17',
        'IN18', 'IN23', 'IN24', 'IN25', 'IN26', 'IN28',
        'IN29', 'IN30', 'IN31', 'IN32', 'IN33', 'IN34',
        'IN35', 'IN38', 'IN39', 'IN40', 'IN41', 'IN42',
        'IN43', 'IN45', 'IN46'
    ]

    for subj in subj_list:
        for i in range(1, 6):
            file_list = glob.glob(
                '/clmnlab/IN/MVPA/LSS_trialtypes/02_deconvolve/%s/LSS.%s.r%02d.*.nii.gz' % (subj, subj, i))
            file_list = sorted(file_list)

            coef_images = []
            tstat_images = []

            for fname in file_list:
                image = mtk.load_5d_fmri_image(fname)
                coef_image = nilearn.image.index_img(imgs=image, index=[1])
                tstat_image = nilearn.image.index_img(imgs=image, index=[2])

                coef_images.append(coef_image)
                tstat_images.append(tstat_image)

            coef_image = nilearn.image.concat_imgs(coef_images)
            tstat_image = nilearn.image.concat_imgs(tstat_images)

            coef_image.to_filename('/clmnlab/IN/MVPA/LSS_trialtypes/03_data/coef.%s.r%02d.nii.gz' % (subj, i))
            tstat_image.to_filename('/clmnlab/IN/MVPA/LSS_trialtypes/03_data/tstat.%s.r%02d.nii.gz' % (subj, i))

            print('%s %02d' % (subj, i), end='\r')