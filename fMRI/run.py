#!/usr/bin/env python3
from ssl import get_default_verify_paths
import sys
import logging
import json
from pathlib import Path
from templateflow.api import get as tpl_get, templates as get_tpl_list
from IPython.display import Image
import nipype.interfaces.io as nio

__version__='1.0.0'
logging.addLevelName(25, 'IMPORTANT')
logging.addLevelName(15, 'VERBOSE')  # Add a new level between INFO and DEBUG
logger = logging.getLogger('cli')

metadata = {
    'Name': 'cat and fox',
    'BIDSVersion': '1.4.1',
    'PipelineDescription': {
        'Name': 'post-fMRIPrep-analysis'
    },
    'OriginalCodeURL': 'https://github.com/poldracklab/ds003-post-fMRIPrep-analysis'
}
   

CONDITION_NAMES = ["Prior" , "Likelihood", "Posterior", "Image_presentation"]
cont01 = ['Prior', 'T', CONDITION_NAMES, [1,0,0,0]]
cont02 = ['Likelihood', 'T', CONDITION_NAMES, [0,1,0,0]]
cont03 = ['Posterior', 'T', CONDITION_NAMES, [0,0,1,0]]
cont04 = ['Image_presentation', 'T', CONDITION_NAMES, [0,0,0,1]]
contrast_list = [cont01,cont02,cont03,cont04]

#define subject group labels
group_label = ['ctl','scz']

def trim_subid(subj_array):
    for ss in range(len(subj_array)):
        #trim 'sub-' prefix
        subj_array[ss] = subj_array[ss][4:]
    return subj_array


def get_parser():
    """Define the command line interface"""
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter

    parser = ArgumentParser(description='DS000003 Analysis Workflow',
                            formatter_class=RawTextHelpFormatter)

    # Arguments as specified by BIDS-Apps
    # required, positional arguments
    # IMPORTANT: they must go directly with the parser object
    parser.add_argument(
        'derivatives_dir', action='store', type=Path,
        help='the root folder of a derivatives set generated with fMRIPrep '
             '(sub-XXXXX folders should be found at the top level in this folder).')
    parser.add_argument('output_dir', action='store', type=Path,
                        help='the output path for the outcomes of preprocessing and visual '
                             'reports')
    parser.add_argument('analysis_level', choices=['run', 'subject','sub_higher', 'group', 'group-diff'], nargs='+',
                        help='processing stage to be run, "run" means run analysis of each subject, "subject" means individual  (combine runs) '
                             ', "group" is second level analysis.')

    parser.add_argument('--version', action='version', version=__version__)

    # Options that affect how pyBIDS is configured
    g_bids = parser.add_argument_group('Options for filtering BIDS queries')
    g_bids.add_argument('--participant-label', action='store', type=str,
                        nargs='*', help='process only particular subjects')
    g_bids.add_argument('--task', action='store', type=str, nargs='*',
                        help='select a specific task to be processed')
    g_bids.add_argument('--run', action='store', type=int, nargs='*',
                        help='select a specific run identifier to be processed')
    g_bids.add_argument('--space', action='store', choices=get_tpl_list() + ['T1w', 'template'],
                        help='select a specific space to be processed')
    g_bids.add_argument('--bids-dir', action='store', type=Path,
                        help='point to the BIDS root of the dataset from which the derivatives '
                             'were calculated (in case the derivatives folder is not the default '
                             '(i.e. ``BIDS_root/derivatives``).')

    g_perfm = parser.add_argument_group('Options to handle performance')
    g_perfm.add_argument("-v", "--verbose", dest="verbose_count", action="count", default=0,
                         help="increases log verbosity for each occurence, debug level is -vvv")
    g_perfm.add_argument('--ncpus', '--nprocs', action='store', type=int,
                         help='maximum number of threads across all processes')
    g_perfm.add_argument('--nthreads', '--omp-nthreads', action='store', type=int,
                         help='maximum number of threads per-process')

    g_other = parser.add_argument_group('Other options')
    g_other.add_argument('-w', '--work-dir', action='store', type=Path,
                         help='path where intermediate results should be stored')
    g_other.add_argument('--fwhm', action='store', type=float, default=6.0,
                         help='FWHM of smoothing')

    return parser


def main():
    """Entry point"""
    from os import cpu_count
    from multiprocessing import set_start_method
    from bids.layout import BIDSLayout
    from nipype import logging as nlogging
    set_start_method('forkserver')
    print('nakaha?')

    opts = get_parser().parse_args()

    # Retrieve logging level
    log_level = int(max(25 - 5 * opts.verbose_count, logging.DEBUG))
    # Set logging
    logger.setLevel(log_level)
    nlogging.getLogger('nipype.workflow').setLevel(log_level)
    nlogging.getLogger('nipype.interface').setLevel(log_level)
    nlogging.getLogger('nipype.utils').setLevel(log_level)

    # Resource management options
    plugin_settings = {
        'plugin': 'MultiProc',
        'plugin_args': {
            'n_procs': opts.ncpus,
            'mem_gb' : 8,
            'raise_insufficient': True,
            'maxtasksperchild': 1,
        }
    }

    # Permit overriding plugin config with specific CLI options
    if not opts.ncpus or opts.ncpus < 1:
        plugin_settings['plugin_args']['n_procs'] = cpu_count()

    nthreads = opts.nthreads
    if not nthreads or nthreads < 1:
        nthreads = cpu_count()

    derivatives_dir = opts.derivatives_dir.resolve()
    bids_dir = opts.bids_dir or derivatives_dir.parent

    # get absolute path to BIDS directory
    bids_dir = opts.bids_dir.resolve()
    print(bids_dir)
    print(str(derivatives_dir))
    layout = BIDSLayout(str(bids_dir), validate=False, derivatives=derivatives_dir) #subject:DI, session:0 , run:5
    query = {'scope': 'derivatives', 'desc': 'preproc',
             'suffix': 'bold', 'extension': ['.nii', '.nii.gz']}
    print(layout)

    if opts.participant_label:
        query['subject'] = '|'.join(opts.participant_label)
    if opts.run:
        query['run'] = '|'.join(opts.run)
    if opts.task:
        query['task'] = '|'.join(opts.task)
    if opts.space:
        query['space'] = opts.space
        if opts.space == 'template':
            query['space'] = get_tpl_list()

    # Preprocessed files that are input to the workflow
    prepped_bold = layout.get(**query)
    if not prepped_bold:
        print('No preprocessed files found under the given derivatives '
              'folder "%s".' % derivatives_dir, file=sys.stderr)

    base_entities = set(['subject', 'session', 'task', 'run', 'acquisition', 'reconstruction'])
    inputs = {}

    for part in prepped_bold:
        entities = part.entities
        sub = entities['subject']
        run = entities['run']
        if not inputs.get(sub):
            inputs[sub] = {}
        inputs[sub][run] = {}
        base = base_entities.intersection(entities)
        subquery = {k: v for k, v in entities.items() if k in base}
        inputs[sub][run]['bold'] = part.path
        inputs[sub][run]['mask'] = layout.get(
            scope='derivatives',
            suffix='mask',
            return_type='file',
            extension=['.nii', '.nii.gz'],
            space=query['space'],
            **subquery)[0]
        inputs[sub][run]['events'] = layout.get(
            suffix='events', return_type='file', **subquery)[0]
        inputs[sub][run]['regressors'] = layout.get(
            scope='derivatives',
            suffix='timeseries',
            return_type='file',
            extension=['.tsv'],
            **subquery)[0]
        inputs[sub][run]['anat'] = layout.get(
            scope='derivatives',
            suffix='T1w',
            return_type='file',
            extension=['.nii', '.nii.gz'],
            space=query['space'],
            subject=subquery['subject'])[0]
        # inputs[sub][run]['anat'] = layout.get(
        #     scope='derivatives',
        #     suffix='T1w',
        #     return_type='file',
        #     extension=['.nii', '.nii.gz'],
        #     subject=subquery['subject'])[0]
        inputs[sub][run]['tr'] = entities['RepetitionTime']

    print(inputs)

    sub_list = sorted(inputs.keys())
    subject = sub_list[0]

    #Determine groupspace
    sample_mask = layout.get(
        scope='derivatives',
        suffix='mask',
        return_type='file',
        extension=['.nii', '.nii.gz'],
        space=query['space'],
        )[0]
    for ss in query['space']:
        if sample_mask.count(ss)==1:
            group_space=ss

    group_t1 = tpl_get(group_space, resolution=1, desc='brain',suffix='T1w')

    bold_tr = prepped_bold[0].entities['RepetitionTime']
    output_dir = opts.output_dir.resolve() #/analysis/afterprep/results

    if 'run' in opts.analysis_level:
        
        from workflows import first_level_wf
        logger.info('Writting 1st level outputs to "%s".', output_dir)

        output_dir_sub = Path(output_dir, subject)
        print(output_dir_sub)
        
        first_level_wf(subject,inputs,output_dir_sub,contrast_list=contrast_list)
        # workflow.base_dir = Path(opts.work_dir,subject)
        # workflow.write_graph(graph2use='flat')
        # workflow.write_graph('workflow_graph.dot')
        # workflow.run(**plugin_settings)

    # if 'subject' in opts.analysis_level:
    #     from workflows import subject_level_wf
    #     logger.info('Writting run level outputs to "%s".', output_dir)

    #     # Total #run for the target subject
    #     total_run = len(inputs[subject])

    #     for cont in range(len(contrast_list)):
    #         cont_name = contrast_list[cont][0]
    #         cont_num = cont+1

    #         # Grabbing cope files of target sub in results of run level analysis
    #         datasource_copes = nio.DataGrabber(infields=['run','cont_num'])
    #         datasource_copes.inputs.base_directory = Path(output_dir,subject)
    #         datasource_copes.inputs.template = 'stats_run/_run_%d/stats/cope%d.nii.gz'
    #         datasource_copes.inputs.run = list(range(1,total_run+1))
    #         datasource_copes.inputs.cont_num = cont_num
    #         datasource_copes.inputs.sort_filelist = True
    #         in_copes = datasource_copes.run().outputs.outfiles

    #         # Grabbing varcope files of target sub in results of run level analysis
    #         datasource_varcopes = nio.DataGrabber(infields=['run','cont_num'])
    #         datasource_varcopes.inputs.base_directory = Path(output_dir,subject)
    #         datasource_varcopes.inputs.template = 'stats_run/_run_%d/stats/varcope%d.nii.gz'
    #         datasource_varcopes.inputs.run = list(range(1,total_run+1))
    #         datasource_varcopes.inputs.cont_num = cont_num
    #         datasource_varcopes.inputs.sort_filelist = True
    #         in_varcopes = datasource_varcopes.run().outputs.outfiles

    #         # Grabbing anatmical file of target sub in fmriprep dir
    #         anat_img = inputs[subject][1]['anat'] # For plot_stats
    #         wf_name = 'wf_subject_level_'+cont_name
    #         output_dir_sub = Path(output_dir, subject)
    #         workflow = subject_level_wf(output_dir=output_dir_sub, bold_tr=bold_tr, anat_img=anat_img, cont_name=cont_name, fwhm=opts.fwhm, name=wf_name)

    #         # set inputs
    #         workflow.inputs.inputnode.group_mask = inputs[subject][1]['mask'] # Use mask of run01
    #         # workflow.inputs.inputnode.group_mask = str(group_mask)
    #         workflow.inputs.inputnode.in_copes = in_copes
    #         workflow.inputs.inputnode.in_varcopes = in_varcopes

    #         workflow.base_dir = Path(opts.work_dir, subject)
    #         workflow.write_graph(graph2use='flat')
    #         workflow.write_graph("workflow_graph.dot")
    #         workflow.run(**plugin_settings)

    if 'group' in opts.analysis_level:
        from workflows import group_level_wf, make_groupmask_wf
        import re

        output_dir = opts.output_dir.resolve()

        #Read participant information
        import pandas as pd
        subjfile = '{}/participants.tsv'.format(bids_dir)
        df_subj = pd.read_table(subjfile, index_col=0)

        #separates subjects in groups
        subj_grouped = {}
        for gl in group_label:
            group_info = df_subj.query('group == @gl')
            subj_grouped[gl] = trim_subid(group_info.index.values)
        print(subj_grouped)

        # make groupmask
        subs = list(inputs.keys())
        ref_file = inputs[subs[0]][1]['mask']
        group_mask = Path(output_dir, 'groupmask', group_space+'_mask.nii.gz')
        group_mask.parent.mkdir(exist_ok=True, parents=True)
        wf_groupmask = make_groupmask_wf(output_file=group_mask, group_space=group_space,ref_file=ref_file,name='wf_make_groupmask')
        wf_groupmask.run(**plugin_settings)

        for cont in range(len(contrast_list)):
            cont_name = contrast_list[cont][0]
            cont_num = cont+1

            # Second loop corresponds to the subject group
            for gl in group_label:
                subj = subj_grouped[gl].tolist()

                # Grabbing cope files in results of 1st level analysis
                datasource_copes = nio.DataGrabber(infields=['subject_id', 'cont_name'])
                datasource_copes.inputs.base_directory = output_dir
                datasource_copes.inputs.template = '%s/stats_subject/%s/cope1.nii.gz'
                datasource_copes.inputs.subject_id = subj #e.g. ['171026Iwasaki', '171218Sakaguchi']
                datasource_copes.inputs.cont_name = cont_name
                datasource_copes.inputs.sort_filelist = True
                print(datasource_copes.inputs)
                grabbed_temp = datasource_copes.run()
                in_copes = grabbed_temp.outputs

                # Grabbing varcope files in results of 1st level analysis
                datasource_varcopes = nio.DataGrabber(infields=['subject_id', 'cont_name'])
                datasource_varcopes.inputs.base_directory = output_dir
                datasource_varcopes.inputs.template = '%s/stats_subject/%s/varcope1.nii.gz'
                datasource_varcopes.inputs.subject_id = subj #e.g. ['171026Iwasaki', '171218Sakaguchi']
                datasource_varcopes.inputs.cont_name = cont_name
                datasource_varcopes.inputs.sort_filelist = True
                grabbed_temp = datasource_varcopes.run()
                in_varcopes = grabbed_temp.outputs

                wf_name = 'wf_group_level'
                output_dir_gl = Path(output_dir, gl+'_'+cont_name)
                anat_img = str(group_t1)
                workflow = group_level_wf(output_dir=output_dir_gl, bold_tr=bold_tr, anat_img=anat_img, cont_name=cont_name, fwhm=opts.fwhm, name=wf_name)

                # set inputs
                workflow.inputs.inputnode.group_mask = group_mask
                workflow.inputs.inputnode.in_copes = in_copes.outfiles
                workflow.inputs.inputnode.in_varcopes = in_varcopes.outfiles

                workflow.base_dir = Path(opts.work_dir, gl+'_'+cont_name)
                workflow.write_graph(graph2use='flat')
                workflow.write_graph("workflow_graph.dot")
                workflow.run(**plugin_settings)

    if 'group-diff' in opts.analysis_level:
        from workflows import group_diff_wf, make_groupmask_wf
        import re

        output_dir = opts.output_dir.resolve()

        #Read participant information
        import pandas as pd
        subjfile = '{}/participants.tsv'.format(bids_dir)
        df_subj = pd.read_table(subjfile, index_col=0)

        #Read participants information
        subj_grouped = {}
        for gl in group_label:
            group_info = df_subj.query('group == @gl')
            subj_grouped[gl] = trim_subid(group_info.index.values)

        group_mask = Path(output_dir, 'groupmask', group_space+'_mask.nii.gz')

        for cont in range(len(contrast_list)):
            cont_name = contrast_list[cont][0]
            cont_num = cont+1

            # Use all subjects
            # (this sort order of subj by group)
            subj = subj_grouped['ctl'].tolist() + subj_grouped['scz'].tolist()

            # Constract a design matrix as 2dlist
            list_design = [ [1,0] for x in range(len(subj_grouped['ctl'])) ] + [ [0,1] for x in range(len(subj_grouped['scz'])) ]

            # Grabbing cope files in results of 1st level analysis
            datasource_copes = nio.DataGrabber(infields=['subject_id', 'cont_name'])
            datasource_copes.inputs.base_directory = output_dir
            datasource_copes.inputs.template = '%s/stats_subject/%s/cope1.nii.gz'
            datasource_copes.inputs.subject_id = subj #e.g. ['171026Iwasaki', '171218Sakaguchi']
            datasource_copes.inputs.cont_name = cont_name
            datasource_copes.inputs.sort_filelist = True
            grabbed_temp = datasource_copes.run()
            in_copes = grabbed_temp.outputs

            # Grabbing varcope files in results of 1st level analysis
            datasource_varcopes = nio.DataGrabber(infields=['subject_id', 'cont_name'])
            datasource_varcopes.inputs.base_directory = output_dir
            datasource_varcopes.inputs.template = '%s/stats_subject/%s/varcope1.nii.gz'
            datasource_varcopes.inputs.subject_id = subj #e.g. ['171026Iwasaki', '171218Sakaguchi']
            datasource_varcopes.inputs.cont_name = cont_name
            datasource_varcopes.inputs.sort_filelist = True
            grabbed_temp = datasource_varcopes.run()
            in_varcopes = grabbed_temp.outputs

            wf_name = 'wf_group_diff'
            output_dir_gl = Path(output_dir, 'groupdiff_'+cont_name)
            anat_img = str(group_t1)
            workflow = group_diff_wf(output_dir=output_dir_gl, bold_tr=bold_tr, anat_img=anat_img, cont_name=cont_name, fwhm=opts.fwhm, name=wf_name)

            # set inputs
            workflow.inputs.inputnode.group_mask = group_mask
            workflow.inputs.inputnode.in_copes = in_copes.outfiles
            workflow.inputs.inputnode.in_varcopes = in_varcopes.outfiles
            workflow.inputs.inputnode.list_design = list_design

            workflow.base_dir = Path(opts.work_dir, 'groupdiff_'+cont_name)
            workflow.write_graph(graph2use='flat')
            workflow.write_graph("workflow_graph.dot")
            workflow.run(**plugin_settings)



if __name__=='__main__':
    print('doukana')
    sys.exit(main())


print("")