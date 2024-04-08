"""
Analysis workflows
"""
from pathlib import Path
import pandas as pd
import numpy as np
from pathlib import Path
import os
from nilearn.image import new_img_like, load_img, get_data, clean_img, concat_imgs, index_img, resample_to_img, mean_img, smooth_img
from nilearn.plotting import plot_glass_brain, plot_img, plot_stat_map, show
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn import datasets
from nilearn.reporting import make_glm_report, get_clusters_table
from sklearn.model_selection import cross_validate
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker, NiftiSpheresMasker
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import nilearn.decoding
from sklearn.model_selection import LeaveOneGroupOut
from bids.layout import BIDSLayout
from nilearn import image as nimg
from nilearn.glm import threshold_stats_img
import seaborn as sns


from sklearn import svm
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import nibabel as nib



def create_design_matrix(sub,run,run_dir):
    tr =2.0
    n_scans = 199
    frame_times = np.arange(n_scans)
    fmriprep_dir = '/bucket/DoyaU/Shuhei/cat_fox/fMRI/fmriprep'
    layout  = BIDSLayout(fmriprep_dir, validate=False,
                            config=['bids','derivatives'])
    confound_files = layout.get(subject=sub,datatype='func',desc='confounds',
                            extension="tsv",return_type='file')

    ##ここに
    # IM_event_dir = '/bucket/DoyaU/Shuhei/cat_fox/fMRI/event/event_IM/sub-'+str(sub)+'/sub-'+str(sub)+'_task-bayes_run-00'+str(run)+'_events.tsv'
    IM_event_dir = '/bucket/DoyaU/Shuhei/cat_fox/fMRI/event/events/sub-'+str(sub)+'/sub-'+str(sub)+'_task-bayes_run-00'+str(run)+'_events.tsv'
    IM_event_file = pd.read_csv(IM_event_dir, delimiter='\t')
    IM_event_file['onset'] = IM_event_file['onset']/2
    IM_event_file['duration'] = IM_event_file['duration']/2
    # IM_event_file = IM_event_file.rename({'amplitudes': 'modulation'},axis='columns')

    # print(event_file)
    frame_times = np.arange(199)

    X2 = make_first_level_design_matrix(
            frame_times,
            IM_event_file,
            drift_model=None,
            hrf_model='spm',
        )
 
    X2 = X2.loc[4:]
    X2 = X2.drop(['constant'],axis=1)
    # X2 = X2.reindex(columns=['Prior','Likelihood','Posterior'])
    X2 = X2.reindex(columns=['Prior','Likelihood','Posterior'])
    # print(X2)

    # event zero duration
    ##ここに
    # IM_event_dir = '/bucket/DoyaU/Shuhei/cat_fox/fMRI/event/event_IM/sub-'+str(sub)+'/sub-'+str(sub)+'_task-bayes_run-00'+str(run)+'_events.tsv'
    # IM_event_dir = '/bucket/DoyaU/Shuhei/cat_fox/fMRI/event/event_zero/sub-'+str(sub)+'/sub-'+str(sub)+'_task-bayes_run-00'+str(run)+'_events.tsv'
    # IM_event_file = pd.read_csv(IM_event_dir, delimiter='\t')
    # IM_event_file['onset'] = IM_event_file['onset']/2
    # IM_event_file['duration'] = IM_event_file['duration']/2
    # IM_event_file = IM_event_file.rename({'amplitudes': 'modulation'},axis='columns')

    # print(event_file)
    # frame_times = np.arange(199)

    # X_zero = make_first_level_design_matrix(
    #         frame_times,
    #         IM_event_file,
    #         drift_model=None,
    #         hrf_model='spm',
    #     )
 
    # X_zero = X_zero.loc[4:]
    # X_zero = X_zero.drop(['constant'],axis=1)
    # # X2 = X2.reindex(columns=['Prior','Likelihood','Posterior'])
    # X_zero = X_zero.reindex(columns=['Pri_zero','Lik_zero','Pos_zero'])
    # # print(X2)
    # X3 = pd.concat([X2,X_zero],axis=1)


    
    arg = run-1
    confound_file = confound_files[arg]
    # Select confounds
    confound_vars = ['trans_x','trans_y','trans_z',
                    'rot_x','rot_y','rot_z']
    derivative_columns = ['a_comp_cor_%02d' % i for i in range(6)]# + ['cosine%02d' % i for i in range(4)]
    # derivative_columns = []
    dis = ['framewise_displacement']
    
    confound_df = pd.read_csv(confound_file, delimiter='\t')
    final_confounds = confound_vars + derivative_columns +dis
    confound_df = confound_df[final_confounds]
    confounds_matrix = confound_df.values
    confounds_matrix = np.nan_to_num(confounds_matrix)

    
    event_dir = '/bucket/DoyaU/Shuhei/cat_fox/fMRI/event/event_timing/sub-'+str(sub)+'/sub-'+str(sub)+'_task-bayes_run-00'+str(run)+'_events.tsv'
    event_file = pd.read_csv(event_dir, delimiter='\t')
    event_file['onset'] = event_file['onset']/2
    # event_file['duration'] = event_file['duration']/2

    X_timing = make_first_level_design_matrix(
        frame_times,
        event_file,
        drift_model=None,
        hrf_model='spm',
        add_regs=confounds_matrix,
        add_reg_names=final_confounds,
    )

    X_timing = X_timing.loc[4:]
    X_timing=X_timing.rename({'Prior':'Cue_timing','Likelihood':'Image_timing','Posterior':'Response_timing'},axis='columns')

    # X_timing = X_timing.drop(['Cue_timing','Response_timing'],axis=1)
    
    
    X = pd.concat([X2,X_timing],axis=1)
    
    plot_design_matrix(X)

    plt.subplots_adjust(left=0.08, top=0.9, bottom=0.21, right=0.96, wspace=0.3)
    plt.savefig(os.path.join(run_dir,f'design_matrix_run{run}.jpg'))
    plt.show()
    
    # plotting correlation heatmap
    dataplot = sns.heatmap(X.corr(), cmap="YlGnBu")
    # displaying heatmap
    plt.savefig(os.path.join(run_dir,f'regression_correlation_run{run}.jpg'))
    # plt.show()

    return X

def clean(sub,run):
    fmriprep_dir = '/bucket/DoyaU/Shuhei/cat_fox/fMRI/fmriprep'
    layout = BIDSLayout(fmriprep_dir,validate=False,
                                config=['bids','derivatives'])

    func_files = layout.get(subject=sub, datatype='func',desc='preproc',
                            space='MNI152NLin2009cAsym',extension='nii.gz',return_type='file')

    mask_files = layout.get(subject=sub,datatype='func',desc='brain',
                            space='MNI152NLin2009cAsym',extension='nii.gz',return_type='file')


    arg = run-1
    func_file = func_files[arg]
    mask_file = mask_files[arg]

    raw_func_img = nimg.load_img(func_file)


    func_img = raw_func_img.slicer[:,:,:,4:]


    # Set some constants
    high_pass = 0.0078
    # low_pass = 0.08
    t_r = 2

    #Clean! without confounds
    clean_img = nimg.clean_img(func_img,detrend=True,standardize=True,
                                high_pass=high_pass,t_r=t_r,mask_img=mask_file)

    # clean_img = smooth_img(clean_img, 6)

    return clean_img, mask_file

DATA_ITEMS = ['bold', 'mask', 'events', 'regressors', 'tr']

def first_level_wf(sub,in_files, output_dir, contrast_list, fwhm=6.0, name='wf_1st_level'):

    os.makedirs(output_dir, exist_ok=True)


    fmri_img = []
    design_matrices = []
    for run in range(1,6):
        run_dir = Path(output_dir , 'run-' + str(run))
        if not Path(run_dir).exists():
            os.makedirs(run_dir)

        X = create_design_matrix(sub,run,run_dir)

        clean_img, mask = clean(sub, run)

        fmri_img.append(clean_img)
        design_matrices.append(pd.DataFrame(X))
    
    fmri_glm = FirstLevelModel(mask_img=mask, minimize_memory=False, signal_scaling=False,smoothing_fwhm=6)
    fmri_glm = fmri_glm.fit(fmri_img,design_matrices=design_matrices)

    n_columns = design_matrices[0].shape[1]

    # check later
    contrasts = {'Prior': pad_vector([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_columns),
            'Likelihood': pad_vector([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_columns),
            'Posterior':     pad_vector([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_columns),
            'Image_timing':  pad_vector([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_columns)}
    # contrasts = {'Image_timing': pad_vector([1, 0, 0, 0, 0, 0, 0, 0, 0], n_columns)}
    # contrast_id = 'Image_timing'
    mean_img_ = mean_img(fmri_img)
    
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        z_map = fmri_glm.compute_contrast(contrasts[contrast_id],output_type='z_score')
        z_map.to_filename(Path(output_dir, f'{contrast_id}_z_map.nii.gz'))

        plot_stat_map(
            z_map, threshold=3.0,
            title=f'{contrast_id}, fixed effects')
        plt.savefig(os.path.join(output_dir,f'{contrast_id}, fixed effect.jpg'))

        clean_map, threshold = threshold_stats_img(
            z_map, alpha=0.001,height_control='fpr',cluster_threshold=10
        )

        plot_stat_map(
            clean_map,
            threshold=threshold,
            # display_mode="z", 
            # cut_coords=3,
            black_bg=True,
            title=f"{contrast_id}(fdr=0.05), clusters > 10 voxels",
        )
        plt.savefig(os.path.join(output_dir, f"{contrast_id}(clusters > 10 voxels).jpg"))

        print(threshold)
        table = get_clusters_table(z_map, stat_threshold=threshold, cluster_threshold=10)
        table.set_index("Cluster ID", drop=True)
        print(table.head())

        # get the 6 largest clusters' max x, y, and z coordinates
        max_row = table.shape[0]
        if max_row > 1:
            coords = table.iloc[range(0, 2)][["X", "Y", "Z"]].values
        else:
            coords = [[23.5,53.5,21.5],[-30,-80,-20]]
    
        # coords = [[23.5,53.5,21.5],[-30,-80,-20]]
        masker = NiftiSpheresMasker(coords)
        real_timeseries = masker.fit_transform(concat_imgs(fmri_img))
        predicted_timeseries = masker.fit_transform(concat_imgs(fmri_glm.predicted))

        # colors for each of the clusters
        colors = ["blue", "navy", "purple", "magenta", "olive", "teal"]
        # plot the time series and corresponding locations
        fig1, axs1 = plt.subplots(2, 2)
        for i in range(2):
            # plotting time series
            axs1[0, i].set_title(f"Cluster peak {coords[i]}\n")
            axs1[0, i].plot(real_timeseries[:, i], c=colors[i], lw=2)
            axs1[0, i].plot(predicted_timeseries[:, i], c="r", ls="--", lw=2)
            axs1[0, i].set_xlabel("Time")
            axs1[0, i].set_ylabel("Signal intensity", labelpad=0)
        #     plotting image below the time series
            roi_img = plot_stat_map(
        #         clean_img,
                clean_map,
                cut_coords=[coords[i][2]],
                threshold=3.1,
                figure=fig1,
                axes=axs1[1, i],
                display_mode="z",
                colorbar=False,
                # bg_img=mean_img_,
            )
            roi_img.add_markers([coords[i]], colors[i], 300)
        fig1.set_size_inches(24, 14)
        plt.savefig(os.path.join(output_dir, f"signal_extraction_{contrast_id}.jpg"))

    # for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):

    #     # fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matrices)
    #     z_map = fmri_glm.compute_contrast(
    #         contrasts[contrast_id], output_type='z_score')
    #     z_map.to_filename(Path(output_dir, f'{contrast_id}_z_map.nii.gz'))
    #     plot_stat_map(
    #         z_map, threshold=3.0,
    #         title=f'{contrast_id}, fixed effects')
    #     plt.savefig(os.path.join(output_dir,f'{contrast_id}, fixed effect.jpg'))

    #     #runごとって必要なのか？　なんか要らないんじゃないか？
    #     # for run in range(1,6):
    #     #     index = run-1
    #     #     fmri_glm = fmri_glm.fit(fmri_img[index], design_matrices=design_matrices[index])
    #     #     z_map = fmri_glm.compute_contrast(
    #     #         contrasts[contrast_id], output_type='z_score')
    #     #     plot_stat_map(
    #     #         z_map, threshold=3.0,
    #     #         bg_img = mean_img_,
    #     #         title=f'{contrast_id}, run {str(run)}')
    #     #     plt.savefig(os.path.join(run_dir,f'run{run}_{contrast_id}.jpg'))

    #     #     beta_map_path = os.path.join(run_dir, f'run{run}_{contrast_id}_z_map.nii.gz') # This is the beta map, right?
    #     #     z_map.to_filename(beta_map_path)

    #     # ここで補正をかける
    #     clean_map, threshold = threshold_stats_img(
    #         z_map, alpha=0.001, height_control="fpr", cluster_threshold=10
    #     )
    #     plot_stat_map(
    #         clean_map,
    #         threshold=threshold,
    #         # display_mode="z", 
    #         # cut_coords=3,
    #         black_bg=True,
    #         title=f"{contrast_id}(fdr=0.05), clusters > 10 voxels",
    #     )
    #     plt.savefig(os.path.join(output_dir, f"{contrast_id}(clusters > 10 voxels).jpg"))

    #     table = get_clusters_table(clean_map, stat_threshold=threshold, cluster_threshold=10)
    #     table.set_index("Cluster ID", drop=True)
    #     print("table: ", table)
    #     max_row = table.shape[0]
    #     if max_row > 1:
    #         coords = table.iloc[range(0, 2)][["X", "Y", "Z"]].values
    #     else:
    #         coords = [[23.5,53.5,21.5],[-30,-80,-20]]


    #     masker = NiftiSpheresMasker(coords)
    #     real_timeseries = masker.fit_transform(concat_imgs(fmri_img))
    #     predicted_timeseries = masker.fit_transform(concat_imgs(fmri_glm.predicted))

    #     # colors for each of the clusters
    #     colors = ["blue", "navy", "purple", "magenta", "olive", "teal"]
    #     # plot the time series and corresponding locations
    #     fig1, axs1 = plt.subplots(2, 2)
    #     for i in range(2):
    #         # plotting time series
    #         axs1[0, i].set_title(f"Cluster peak {coords[i]}\n")
    #         axs1[0, i].plot(real_timeseries[:, i], c=colors[i], lw=2)
    #         axs1[0, i].plot(predicted_timeseries[:, i], c="r", ls="--", lw=2)
    #         axs1[0, i].set_xlabel("Time")
    #         axs1[0, i].set_ylabel("Signal intensity", labelpad=0)
    #     #     plotting image below the time series
    #         roi_img = plot_stat_map(
    #     #         clean_img,
    #             clean_map,
    #             cut_coords=[coords[i][2]],
    #             threshold=threshold,
    #             figure=fig1,
    #             axes=axs1[1, i],
    #             display_mode="z",
    #             colorbar=False,
    #             # bg_img=mean_img_,
    #         )
    #         roi_img.add_markers([coords[i]], colors[i], 300)
    #     fig1.set_size_inches(24, 14)
    #     plt.savefig(os.path.join(output_dir,f'{contrast_id}, signal_extraction.jpg'))


    report = make_glm_report(fmri_glm,
                             contrasts,
                             cluster_threshold=10
                             )
    report.save_as_html(os.path.join(output_dir,'report.html'))

    return


def group_level_wf(output_dir, bold_tr, anat_img, cont_name, fwhm=6.0, name='wf_group_level'):
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['group_mask', 'in_copes', 'in_varcopes']),
        name='inputnode')

    # Configure FSL 2nd level analysis
    # L2Model just generates design matrix for one-sample T-test
    # (For two-sample T-test across groups, we need to make design file by ourself)
    l2_model = pe.Node(fsl.L2Model(), name='l2_model')

    # Set run_mode from ['fe' (fixed effect) or 'ols' or 'flame1' or 'flame12']
    # Details are on https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FEAT/UserGuide#Setting_Up_Higher-Level_Analysis_in_FEAT
    # Simply, accuracy is flame12>flame1>ols, but computation is ols>flame1>flame12
    flameo_ols = pe.Node(fsl.FLAMEO(run_mode='flame1'), name='flameo_ols')

    # Merge cope and varcope files across subjects
    # Concat direction is t=time
    merge_copes = pe.Node(fsl.Merge(dimension='t', tr=bold_tr), name='merge_copes')
    merge_varcopes = pe.Node(fsl.Merge(dimension='t', tr=bold_tr), name='merge_varcopes')

    # Thresholding - FDR ################################################
    # Calculate pvalues with ztop
    fdr_ztop = pe.Node(fsl.ImageMaths(op_string='-ztop', suffix='_pval'),
                       name='fdr_ztop')


    # Thresholding - FWE ################################################
    # smoothest -r %s -d %i -m %s
    smoothness = pe.Node(fsl.SmoothEstimate(), name='smoothness')
    # ptoz 0.025 -g %f
    # p = 0.05 / 2 for 2-tailed test
    fwe_ptoz = pe.Node(PtoZ(pvalue=0.025), name='fwe_ptoz')
    # fslmaths %s -uthr %s -thr %s nonsignificant
    # fslmaths %s -sub nonsignificant zstat1_thresh
    fwe_nonsig0 = pe.Node(fsl.Threshold(direction='above'), name='fwe_nonsig0')
    fwe_nonsig1 = pe.Node(fsl.Threshold(direction='below'), name='fwe_nonsig1')
    fwe_thresh = pe.Node(fsl.BinaryMaths(operation='sub'), name='fwe_thresh')



    zstat_inv = pe.Node(fsl.BinaryMaths(operation='mul', operand_value=-1),
                        name='zstat_inv')

    # Plot zmap
    # plot_zmap = pe.Node(niu.Function(input_names=['stat_img', 'anat_img', 'output_file', 'thresh', 'fwhm', 'cont_name'],
    #                      function=plot_stats),
    #                      name='plot_zmap')
    # plot_zmap.inputs.fwhm = fwhm
    # plot_zmap.inputs.anat_img = anat_img
    # plot_zmap.inputs.cont_name = cont_name
    # plot_zmap.inputs.thresh = 1
    # plot_zmap.inputs.output_file = Path(output_dir, 'stats_group', cont_name+'.png')

    # Plot fwe (thresholded)
    # plot_fwe = pe.Node(niu.Function(input_names=['stat_img', 'anat_img', 'output_file', 'thresh', 'fwhm', 'cont_name'],
    #                      function=plot_stats),
    #                      name='plot_fwe')
    # plot_fwe.inputs.fwhm = fwhm
    # plot_fwe.inputs.anat_img = anat_img
    # plot_fwe.inputs.cont_name = cont_name
    # plot_fwe.inputs.thresh = 0
    # plot_fwe.inputs.output_file = Path(output_dir, 'stats_group', cont_name+'_fwe.png')

    # Plot cluster
    # plot_clst = pe.Node(niu.Function(input_names=['stat_img', 'anat_img', 'output_file', 'thresh', 'fwhm', 'cont_name'],
    #                      function=plot_stats),
    #                      name='plot_clst')
    # plot_clst.inputs.fwhm = fwhm
    # plot_clst.inputs.anat_img = anat_img
    # plot_clst.inputs.cont_name = cont_name
    # plot_clst.inputs.thresh = 0
    # plot_clst.inputs.output_file = Path(output_dir, 'stats_group', cont_name+'_clst.png')


    ds_zraw = pe.Node(DataSink(
        base_directory=str(output_dir)), name='ds_zraw')

    ds_zfwe = pe.Node(DataSink(
        base_directory=str(output_dir)), name='ds_zfwe')

    ds_zclust = pe.Node(DataSink(
        base_directory=str(output_dir)), name='ds_zclust')

    # ds_clustidx_pos = pe.Node(DataSink(
    #     base_directory=str(output_dir)), name='ds_clustidx_pos')

    # ds_clustlmax_pos = pe.Node(DataSink(
    #     base_directory=str(output_dir)), name='ds_clustlmax_pos')

    # ds_clustidx_neg = pe.Node(DataSink(
    #     base_directory=str(output_dir)), name='ds_clustidx_neg')

    # ds_clustlmax_neg = pe.Node(DataSink(
    #     base_directory=str(output_dir)), name='ds_clustlmax_neg')

    workflow.connect([
        (inputnode, l2_model, [(('in_copes', _len), 'num_copes')]),
        (inputnode, flameo_ols, [('group_mask', 'mask_file')]),
        (inputnode, smoothness, [('group_mask', 'mask_file'),
                                 (('in_copes', _dof), 'dof')]),
        (inputnode, merge_copes, [('in_copes', 'in_files')]),
        (inputnode, merge_varcopes, [('in_varcopes', 'in_files')]),

        (l2_model, flameo_ols, [('design_mat', 'design_file'),
                                ('design_con', 't_con_file'),
                                ('design_grp', 'cov_split_file')]),
        (merge_copes, flameo_ols, [('merged_file', 'cope_file')]),
        (merge_varcopes, flameo_ols, [('merged_file', 'var_cope_file')]),
        (flameo_ols, smoothness, [('res4d', 'residual_fit_file')]),

        (flameo_ols, fwe_nonsig0, [('zstats', 'in_file')]),
        (fwe_nonsig0, fwe_nonsig1, [('out_file', 'in_file')]),
        (smoothness, fwe_ptoz, [('resels', 'resels')]),
        (fwe_ptoz, fwe_nonsig0, [('zstat', 'thresh')]),
        (fwe_ptoz, fwe_nonsig1, [(('zstat', _neg), 'thresh')]),
        (flameo_ols, fwe_thresh, [('zstats', 'in_file')]),
        (fwe_nonsig1, fwe_thresh, [('out_file', 'operand_file')]),

        (flameo_ols, zstat_inv, [('zstats', 'in_file')]),

        # (flameo_ols, plot_zmap, [('zstats', 'stat_img')]),
        # (fwe_thresh, plot_fwe, [('out_file', 'stat_img')]),

        (flameo_ols, ds_zraw, [('zstats', 'stats_group')]),
        (fwe_thresh, ds_zfwe, [('out_file', 'stats_group')]),
    ])
    return workflow


def group_diff_wf(output_dir, bold_tr, anat_img, cont_name, fwhm=6.0, name='wf_group_diff'):
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['group_mask', 'in_copes', 'in_varcopes', 'list_design']),
        name='inputnode')

    # outputnode = pe.Node(niu.IdentityInterface(
    #    fields=['zstats_raw', 'zstats_fwe', 'zstats_clust',
    #            'clust_index_file', 'clust_localmax_txt_file']),
    #    name='outputnode')

    # Configure FSL 2nd level analysis
    # L2Model just generates design matrix for one-sample T-test
    # (For two-sample T-test across groups, we need to make design file by ourself)
    l2_model = pe.Node(L2diffModel(), name='l2_model')

    # Set run_mode from ['fe' (fixed effect) or 'ols' or 'flame1' or 'flame12']
    # Details are on https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FEAT/UserGuide#Setting_Up_Higher-Level_Analysis_in_FEAT
    # Simply, accuracy is flame12>flame1>ols, but computation is ols>flame1>flame12
    flameo_ols = pe.Node(fsl.FLAMEO(run_mode='flame1'), name='flameo_ols')

    # Merge cope and varcope files across subjects
    # Concat direction is t=time
    merge_copes = pe.Node(fsl.Merge(dimension='t', tr=bold_tr), name='merge_copes')
    merge_varcopes = pe.Node(fsl.Merge(dimension='t', tr=bold_tr), name='merge_varcopes')

    # Thresholding - FDR ################################################
    # Calculate pvalues with ztop
    fdr_ztop = pe.Node(fsl.ImageMaths(op_string='-ztop', suffix='_pval'),
                       name='fdr_ztop')
    # Find FDR threshold: fdr -i zstat1_pval -m <group_mask> -q 0.05
    # fdr_th = <write Nipype interface for fdr>
    # Apply threshold:
    # fslmaths zstat1_pval -mul -1 -add 1 -thr <fdr_th> -mas <group_mask> \
    #     zstat1_thresh_vox_fdr_pstat1

    # Thresholding - FWE ################################################
    # smoothest -r %s -d %i -m %s
    smoothness = pe.Node(fsl.SmoothEstimate(), name='smoothness')
    # ptoz 0.025 -g %f
    # p = 0.05 / 2 for 2-tailed test
    fwe_ptoz = pe.Node(PtoZ(pvalue=0.025), name='fwe_ptoz')
    # fslmaths %s -uthr %s -thr %s nonsignificant
    # fslmaths %s -sub nonsignificant zstat1_thresh
    fwe_nonsig0 = pe.Node(fsl.Threshold(direction='above'), name='fwe_nonsig0')
    fwe_nonsig1 = pe.Node(fsl.Threshold(direction='below'), name='fwe_nonsig1')
    fwe_thresh = pe.Node(fsl.BinaryMaths(operation='sub'), name='fwe_thresh')

    # Thresholding - Cluster ############################################
    # cluster -i %s -c %s -t 3.2 -p 0.025 -d %s --volume=%s  \
    #     --othresh=thresh_cluster_fwe_zstat1 --connectivity=26 --mm
    # cluster_kwargs = {
    #     'connectivity': 26,
    #     'threshold': 3.2,
    #     'pthreshold': 0.025,
    #     'out_threshold_file': True,
    #     'out_index_file': True,
    #     'out_localmax_txt_file': True
    # }
    # cluster_pos = pe.Node(fsl.Cluster(
    #         **cluster_kwargs),
    #     name='cluster_pos')
    # cluster_neg = pe.Node(fsl.Cluster(
    #         **cluster_kwargs),
    #     name='cluster_neg')
    zstat_inv = pe.Node(fsl.BinaryMaths(operation='mul', operand_value=-1),
                        name='zstat_inv')
    # cluster_inv = pe.Node(fsl.BinaryMaths(operation='mul', operand_value=-1),
    #                       name='cluster_inv')
    # cluster_all = pe.Node(fsl.BinaryMaths(operation='add'), name='cluster_all')

    # Plot zmap
    plot_zmap = pe.Node(niu.Function(input_names=['stat_img', 'anat_img', 'output_file', 'thresh', 'fwhm', 'cont_name'],
                         function=plot_stats),
                         name='plot_zmap')
    plot_zmap.inputs.fwhm = fwhm
    plot_zmap.inputs.anat_img = anat_img
    plot_zmap.inputs.cont_name = cont_name
    plot_zmap.inputs.thresh = 1
    plot_zmap.inputs.output_file = Path(output_dir, 'stats_group', cont_name+'.png')

    # Plot fwe (thresholded)
    plot_fwe = pe.Node(niu.Function(input_names=['stat_img', 'anat_img', 'output_file', 'thresh', 'fwhm', 'cont_name'],
                         function=plot_stats),
                         name='plot_fwe')
    plot_fwe.inputs.fwhm = fwhm
    plot_fwe.inputs.anat_img = anat_img
    plot_fwe.inputs.cont_name = cont_name
    plot_fwe.inputs.thresh = 0
    plot_fwe.inputs.output_file = Path(output_dir, 'stats_group', cont_name+'_fwe.png')

    # Plot cluster
    plot_clst = pe.Node(niu.Function(input_names=['stat_img', 'anat_img', 'output_file', 'thresh', 'fwhm', 'cont_name'],
                         function=plot_stats),
                         name='plot_clst')
    plot_clst.inputs.fwhm = fwhm
    plot_clst.inputs.anat_img = anat_img
    plot_clst.inputs.cont_name = cont_name
    plot_clst.inputs.thresh = 0
    plot_clst.inputs.output_file = Path(output_dir, 'stats_group', cont_name+'_clst.png')


    ds_zraw = pe.Node(DataSink(
        base_directory=str(output_dir)), name='ds_zraw')

    ds_zfwe = pe.Node(DataSink(
        base_directory=str(output_dir)), name='ds_zfwe')

    ds_zclust = pe.Node(DataSink(
        base_directory=str(output_dir)), name='ds_zclust')

    ds_clustidx_pos = pe.Node(DataSink(
        base_directory=str(output_dir)), name='ds_clustidx_pos')

    ds_clustlmax_pos = pe.Node(DataSink(
        base_directory=str(output_dir)), name='ds_clustlmax_pos')

    ds_clustidx_neg = pe.Node(DataSink(
        base_directory=str(output_dir)), name='ds_clustidx_neg')

    ds_clustlmax_neg = pe.Node(DataSink(
        base_directory=str(output_dir)), name='ds_clustlmax_neg')

    workflow.connect([
        (inputnode, l2_model, [(('in_copes', _len), 'num_copes'),
                                  ('list_design', 'list_design')]),
        (inputnode, flameo_ols, [('group_mask', 'mask_file')]),
        (inputnode, smoothness, [('group_mask', 'mask_file'),
                                 (('in_copes', _dof), 'dof')]),
        (inputnode, merge_copes, [('in_copes', 'in_files')]),
        (inputnode, merge_varcopes, [('in_varcopes', 'in_files')]),

        (l2_model, flameo_ols, [('design_mat', 'design_file'),
                                ('design_con', 't_con_file'),
                                ('design_grp', 'cov_split_file')]),
        (merge_copes, flameo_ols, [('merged_file', 'cope_file')]),
        (merge_varcopes, flameo_ols, [('merged_file', 'var_cope_file')]),
        (flameo_ols, smoothness, [('res4d', 'residual_fit_file')]),

        (flameo_ols, fwe_nonsig0, [('zstats', 'in_file')]),
        (fwe_nonsig0, fwe_nonsig1, [('out_file', 'in_file')]),
        (smoothness, fwe_ptoz, [('resels', 'resels')]),
        (fwe_ptoz, fwe_nonsig0, [('zstat', 'thresh')]),
        (fwe_ptoz, fwe_nonsig1, [(('zstat', _neg), 'thresh')]),
        (flameo_ols, fwe_thresh, [('zstats', 'in_file')]),
        (fwe_nonsig1, fwe_thresh, [('out_file', 'operand_file')]),

        # (flameo_ols, cluster_pos, [('zstats', 'in_file')]),
        # (merge_copes, cluster_pos, [('merged_file', 'cope_file')]),
        # (smoothness, cluster_pos, [('volume', 'volume'),
        #                            ('dlh', 'dlh')]),
        (flameo_ols, zstat_inv, [('zstats', 'in_file')]),
        # (zstat_inv, cluster_neg, [('out_file', 'in_file')]),
        # (cluster_neg, cluster_inv, [('threshold_file', 'in_file')]),
        # (merge_copes, cluster_neg, [('merged_file', 'cope_file')]),
        # (smoothness, cluster_neg, [('volume', 'volume'),
        #                            ('dlh', 'dlh')]),
        # (cluster_pos, cluster_all, [('threshold_file', 'in_file')]),
        # (cluster_inv, cluster_all, [('out_file', 'operand_file')]),

        (flameo_ols, plot_zmap, [('zstats', 'stat_img')]),
        (fwe_thresh, plot_fwe, [('out_file', 'stat_img')]),
        # (cluster_all, plot_clst, [('out_file', 'stat_img')]),

        (flameo_ols, ds_zraw, [('zstats', 'stats_group')]),
        (fwe_thresh, ds_zfwe, [('out_file', 'stats_group')]),
        # (cluster_all, ds_zclust, [('out_file', 'stats_group')]),
        # (cluster_pos, ds_clustidx_pos, [('index_file', 'stats_group')]),
        # (cluster_pos, ds_clustlmax_pos, [('localmax_txt_file', 'stats_group')]),
        # (cluster_neg, ds_clustidx_neg, [('index_file', 'stats_group')]),
        # (cluster_neg, ds_clustlmax_neg, [('localmax_txt_file', 'stats_group')]),
    ])
    return workflow



def make_groupmask_wf(output_file, group_space, ref_file, name='wf_make_groupmask'):
    from templateflow.api import get as tpl_get, templates as get_tpl_list

    # First, get mask in specified space from templateflow repository
    # This will be downsampled to the input bold resolution
    group_mask = tpl_get(group_space, resolution=1, desc='brain', suffix='mask')

    workflow = pe.Workflow(name=name)

    #Note: ref_file must be the same sapce as template mask
    lineartrans = pe.Node(fsl.FLIRT(), name='lineartrans')
    lineartrans.inputs.in_file = str(group_mask)
    lineartrans.inputs.reference = ref_file
    lineartrans.inputs.output_type = 'NIFTI_GZ'
    lineartrans.inputs.no_resample_blur = True
    lineartrans.inputs.apply_xfm = True
    lineartrans.inputs.uses_qform = True

    # Since resampled mask is not binary, binarize it
    binarization = pe.Node(fsl.UnaryMaths(operation='bin'), name='binarization')
    binarization.inputs.output_type = 'NIFTI_GZ'

    ds_groupmask = pe.Node(ExportFile(), name='ds_groupmask')
    ds_groupmask.inputs.out_file = output_file

    workflow.connect([
        (lineartrans, binarization, [('out_file', 'in_file')]),
        (binarization, ds_groupmask, [('out_file', 'in_file')]),
    ])

    return workflow


def _bids2nipypeinfo(in_file, events_file,regressors_file,
                     regressors_names=None,
                     motion_columns=None,
                     decimals=3, amplitude=1.0):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from nipype.interfaces.base.support import Bunch

    # Process the events file
    events = pd.read_csv(events_file, sep=r'\s+')

    bunch_fields = ['onsets', 'durations', 'amplitudes']

    if not motion_columns:
        from itertools import product
        motion_columns = ['_'.join(v) for v in product(('trans', 'rot'), 'xyz')]

    out_motion = Path('motion.par').resolve()

    regress_data = pd.read_csv(regressors_file, sep=r'\s+')
    np.savetxt(out_motion, regress_data[motion_columns].values, '%g')
    # if regressors_names is None:
    #     regressors_names = sorted(set(regress_data.columns) - set(motion_columns))

    if regressors_names:
        bunch_fields += ['regressor_names']
        bunch_fields += ['regressors']
    


    cond = ['Prior', 'Likelihood', 'Posterior', 'Prior_timing', 'Likelihood_timing', 'Posterior_timing']
    runinfo = Bunch(
        scans=in_file,
        # conditions=list(set(events.trial_type.values)),
        conditions=cond,
        **{k: [] for k in bunch_fields}
        )

    print('runinfo conditions: ',runinfo.conditions)
    for condition in runinfo.conditions:
        if condition =='Prior' or condition == 'Likelihood' or condition == 'Posterior':
            print('ocndition', condition)
            event = events[events.trial_type.str.match(condition)]
            runinfo.onsets.append(np.round(event.onset.values, 3).tolist())
            runinfo.durations.append(np.round(event.duration.values, 3).tolist())
            runinfo.amplitudes.append(np.round(event.amplitudes.values, 3).tolist())
        else:
            print('condition', condition)
            event = events[events.trial_type.str.match(condition[:-7])]
            runinfo.onsets.append(np.round(event.onset.values, 3).tolist())
            runinfo.durations.append(np.round(event.duration.values, 3).tolist())
            runinfo.amplitudes.append([amplitude] * len(event))

    if 'regressor_names' in bunch_fields:
        runinfo.regressor_names = regressors_names
        try:
            runinfo.regressors = regress_data[regressors_names]
        except KeyError:
            regressors_names = list(set(regressors_names).intersection(
                                    set(regress_data.columns)))
            runinfo.regressors = regress_data[regressors_names]
        runinfo.regressors = runinfo.regressors.fillna(0.0).values.T.tolist()


    return [runinfo], str(out_motion)


def _get_tr(in_dict):
    return in_dict.get('RepetitionTime')


def _len(inlist):
    return len(inlist)


def _dof(inlist):
    return len(inlist) - 1


def _neg(val):
    return -val


def _dict_ds(in_dict, sub,run, order=['bold', 'mask', 'events', 'regressors', 'tr']):
    return tuple([in_dict[sub][run][k] for k in order])

# Check whether all keys of all subjects are fullfilled in in_files
def check_infiles(in_files, keys=['bold', 'mask', 'events', 'regressors', 'tr']):
    subjects = in_files.keys()
    errmsg = []
    errtxt = 'Sub {} does not have {}'
    for sub in subjects:
        total_run = len(in_files[sub])
        for run in range(1,total_run+1):
            for k in keys:
                if k not in in_files[sub][run]:
                    errmsg += [errtxt.format(sub, k)]

    if len(errmsg)!=0:
        print('\n'.join(errmsg))
        # Error abort will be added here

    return

def pad_vector(contrast_, n_columns):
    """Append zeros in contrast vectors."""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))