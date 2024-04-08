import pandas as pd
import os

import nibabel as nib
from nilearn.glm.second_level import SecondLevelModel

from nilearn.reporting import make_glm_report
from nilearn import plotting 
from nilearn.glm import threshold_stats_img

from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt

n_subjects = 21
label = ['DI', 'HM', 'RM','KH', 'MF','FA', 'MOt','KT','SY','TY',
'HK','NK','SK','TN','MY', 'NKu','HH', 'MN', 'MYa','TK', 'YA']
subjects_label = [f"sub-{i}" for i in label]


design_matrix = pd.DataFrame(
    [1] * len(label), columns=["group"]
)
design_matrix.loc[0:9,'group'] = -1
design_matrix.loc[10:21,'group'] = 1


def fpr_cluster(z_map,contrast):
    display = plotting.plot_stat_map(z_map,title='Raw z map')
    thresholded_map1, threshold1 = threshold_stats_img(
        z_map,
        alpha=0.005,
        height_control="fpr",
        cluster_threshold=10,
        two_sided=True,
    )


    plotting.plot_stat_map(
        thresholded_map1,
        cut_coords=display.cut_coords,
        threshold=threshold1,
        title="Thresholded z map, fpr <.005, clusters > 10 voxels",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(group_diff_dir,f'{contrast}(fpr,cluster).jpg'))

def fdr(z_map,contrast):
    display = plotting.plot_stat_map(z_map,title='Raw z map')
    thresholded_map2, threshold2 = threshold_stats_img(
        z_map, alpha=0.05, height_control="fdr"
    )
    print(f"The FDR=.05 threshold is {threshold2:.3g}")

    plotting.plot_stat_map(
        thresholded_map2,
        cut_coords=display.cut_coords,
        title="Thresholded z map, expected fdr = .05",
        threshold=threshold2,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(group_diff_dir,f'{contrast}(fdr).jpg'))

def bonferroni(z_map,contrast):
    display = plotting.plot_stat_map(z_map,title='Raw z map')
    thresholded_map3, threshold3 = threshold_stats_img(
        z_map, alpha=0.05, height_control="bonferroni"
    )
    print(f"The p<.05 Bonferroni-corrected threshold is {threshold3:.3g}")
    plotting.plot_stat_map(
        thresholded_map3,
        cut_coords=display.cut_coords,
        title=f"Thresholded z map ({contrast}), expected fwer < .05",
        threshold=threshold3,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(group_diff_dir,f'{contrast}(bonferroni).jpg'))

# group_diff_dir = '/home/s/shuhei-hara1/workspace/conference/figures'

input_dir = '/bucket/DoyaU/Shuhei/cat_fox/fMRI/GLM/cpsy_tokyo'
group_diff_dir = '/flash/DoyaU/shuhei/GLM/cpsy_tokyo/group-diff_con'
os.makedirs(group_diff_dir, exist_ok=True)


cmap_prior = []
cmap_likelihood = []
cmap_posterior = []
cmap_timing = []
for sub in label:
    for contrast in ['Prior','Likelihood','Posterior','Image_timing']:
    # contrast='Image_timing'
        image_file =  f'{input_dir}/{sub}/{contrast}_z_map.nii.gz'
        img = nib.load(image_file)
    # cmap_timing.append(img)
        if contrast == 'Prior':
            cmap_prior.append(img)
        elif contrast == 'Likelihood':
            cmap_likelihood.append(img)
        elif contrast == 'Posterior':
            cmap_posterior.append(img)
        elif contrast == 'Image_timing':
            cmap_timing.append(img)

for contrast in ['Prior', 'Likelihood', 'Posterior', 'Image_timing']:
    if contrast == 'Prior':
        cmap = cmap_prior
    elif contrast == 'Likelihood':
        cmap = cmap_likelihood
    elif contrast == 'Posterior':
        cmap = cmap_posterior
    elif contrast == 'Image_timing':
        cmap = cmap_timing


    # cmap = cmap_timing
    second_level_model = SecondLevelModel(smoothing_fwhm=3)
    second_level_model = second_level_model.fit(
        cmap, design_matrix=design_matrix
    )

    z_map = second_level_model.compute_contrast(output_type='z_score')
    plotting.plot_stat_map(
        z_map, threshold=3.0,
        title=f'{contrast}')
    plt.savefig(os.path.join(group_diff_dir,f'{contrast}_raw_z.jpg'))

    z_image_path = os.path.join(group_diff_dir, f'{contrast}_z_map.nii.gz')
    z_map.to_filename(z_image_path)

    # fpr_cluster(z_map,contrast)
    # fdr(z_map,contrast)
    # bonferroni(z_map,contrast)

    number_cluster = 10
    height_control = 'fpr'
    alpha=0.005
    thresholded_map, threshold = threshold_stats_img(
        z_map, 
        alpha=alpha, #p-value
        height_control = height_control,
        two_sided=True,
        cluster_threshold=number_cluster
    )
    # print(threshold)

    report = make_glm_report(model=second_level_model,contrasts=['group'], threshold=threshold, cluster_threshold=number_cluster, height_control=height_control,alpha=alpha)
    report.save_as_html(os.path.join(group_diff_dir,f'report_{contrast}.html'))


    # report = make_glm_report(model=second_level_model,contrasts=['group'])
    # report.save_as_html(os.path.join(group_diff_dir,f'report_{contrast}.html'))

