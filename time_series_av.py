# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:37:49 2024

@author: bansals3
"""

import mne
import os, os.path as op
import numpy as np
import glob
import pickle
import copy 
from mne.beamformer import make_lcmv, apply_lcmv, apply_lcmv_epochs, apply_lcmv_raw
from mne.datasets import fetch_fsaverage
import pandas as pd
from mne.beamformer import apply_lcmv_cov    
import nibabel as nb
import mne_bids
from mne_bids import BIDSPath
import pathlib

n_jobs = 1 #CHANGE

topdir = pathlib.PurePath('X:\\') # topdir = '/data/ML_MEG'
resultsdir = pathlib.PurePath(op.join(topdir, 'results_s_final'))
#os.chdir(topdir)

#stcdir = op.join(topdir, resultsdir, f'f_{fmin}_{fmax}')
#lh_stc = mne.read_source_estimate('ON25658_stc.h5-lh.stc')
#rh_stc = mne.read_source_estimate('ON25658_stc.h5-rh.stc')


subj_dir = topdir.joinpath('NIH_hvmeg', 'derivatives', 'freesurfer', 'subjects')

#Setup for fs average warping
subjects_dir = subj_dir
fetch_fsaverage(subjects_dir, verbose=False)  # ensure fsaverage src exists
fname_fs_src = subjects_dir.joinpath('fsaverage','bem','fsaverage-ico-5-src.fif')
src_fs = mne.read_source_spaces(fname_fs_src)


def average_stcs(stcs):
    stcs_copy=copy.deepcopy(stcs)

    #Initialize new source estimate for average
    stc_ave = copy.deepcopy(stcs_copy[0])
  
    #Convert to list of numpy arrays
    tmp = [i.data for i in stcs_copy]

    #Stack list into new dimension and average over this epoch dimension
    tmp2 = np.stack(tmp).mean(axis=0)
    stc_ave.data=tmp2
    return stc_ave

def fsaverage_morph(stc):
    src = get_mri(stc.subject[4:])['src']
    morph = mne.compute_source_morph(
        src,
        subject_from='sub-'+stc.subject[4:],
        src_to=src_fs,
        subjects_dir=subj_dir,
        niter_sdr=[5, 5, 3],
        niter_affine=[100, 100, 10],
        zooms=3,  # just for speed
        verbose=True,
    )
    stc_morph = morph.apply(stc)
    return stc_morph



def get_mri(subject):
    if subject[0:2] == 'ON':
        #subject = 'ON02747'    
        bids_dir = topdir.joinpath('NIH_hvmeg')#'X:\\NIH_hvmeg'
        output_dirname = 'sternberg_res_3_6_final'      ###RESET NAME!!!!
        task_type = 'sternberg'
        # =============================================================================
        # general variables
        # =============================================================================
        deriv_dir = op.join(bids_dir, 'derivatives')
        output_dir = op.join(deriv_dir, output_dirname)
        subjects_dir = op.join(deriv_dir, 'freesurfer', 'subjects')
        os.environ['SUBJECTS_DIR']=subjects_dir
    
        bids_path = BIDSPath(root=bids_dir, subject=subject, datatype='meg',
                             task=task_type, session ='01', run = '01')
        anat_bids_path = BIDSPath(root=bids_dir, subject=subject, datatype='anat',
                                  extension='.nii', acquisition = 'MPRAGE', suffix = 'T1w', session = '01')
        '''if os.path.exists(anat_bids_path) == False:
            anat_bids_path = BIDSPath(root=bids_dir, subject=subject, datatype='anat',
                                      extension='.nii', acquisition = 'FSPGR', suffix = 'T1w', session = '01', recording='SCIC')
        else:
            pass'''
        deriv_path = bids_path.copy().update(root=output_dir, check=False)
        deriv_path.directory.mkdir(exist_ok=True, parents=True)
    
        raw_fname = bids_path.copy().update(run = '01', session = '01')
        bem_fname = deriv_path.copy().update(suffix='bem', extension='.fif')
        fwd_fname = deriv_path.copy().update(suffix='fwd', extension='.fif')
        src_fname = deriv_path.copy().update(suffix='src', extension='.fif')
        trans_fname = deriv_path.copy().update(suffix='trans',extension='.fif')
        raw = mne.io.read_raw_ctf(raw_fname.fpath, system_clock = 'ignore', clean_names =True)
        
        subjects_dir = mne_bids.read.get_subjects_dir()
        fs_subject = 'sub-'+bids_path.subject
        if not bem_fname.fpath.exists():
            mne.bem.make_watershed_bem(fs_subject, subjects_dir=subjects_dir, overwrite=True)
            bem = mne.make_bem_model(fs_subject, subjects_dir=f'{subjects_dir}', 
                                     conductivity=[0.3])
            bem_sol = mne.make_bem_solution(bem)
            
            mne.write_bem_solution(bem_fname, bem_sol, overwrite=True)
        else:
            bem_sol = mne.read_bem_solution(bem_fname)
            
        if not src_fname.fpath.exists():
            src = mne.setup_source_space(fs_subject, spacing='oct6', add_dist='patch',
                                 subjects_dir=subjects_dir)
            src.save(src_fname.fpath, overwrite=True)
        else:
            src = mne.read_source_spaces(src_fname.fpath)
    
        if not trans_fname.fpath.exists():
            trans = mne_bids.read.get_head_mri_trans(bids_path, extra_params=dict(system_clock='ignore'),
                                                t1_bids_path=anat_bids_path, fs_subject=fs_subject, 
                                                fs_subjects_dir=subjects_dir)
            mne.write_trans(trans_fname.fpath, trans, overwrite=True)
        else:
            trans = mne.read_trans(trans_fname.fpath)
        #if fwd_fname.fpath.exists():
          #  fwd = mne.read_forward_solution(fwd_fname)
        #else:
        fwd = mne.make_forward_solution(raw.info, trans, src, bem_sol, eeg=False, 
                                        n_jobs=n_jobs)
        mne.write_forward_solution(fwd_fname.fpath, fwd, overwrite=True)
        return {'fwd': fwd, 'trans': trans, 'src': src}
    else:
        #subject = 'ON02747'    
        bids_dir = topdir.joinpath('NIH_hvmeg')#'X:\\NIH_hvmeg'
        output_dirname = 'sternberg_res_3_6_final'      ###RESET NAME!!!!
        task_type = 'sternberg'
        # =============================================================================
        # general variables
        # =============================================================================
        deriv_dir = op.join(bids_dir, 'derivatives')
        output_dir = op.join(deriv_dir, output_dirname)
        subjects_dir = op.join(deriv_dir, 'freesurfer', 'subjects')
        os.environ['SUBJECTS_DIR']=subjects_dir
    
        bids_path = BIDSPath(root=bids_dir, subject=subject, datatype='meg',
                             task=task_type, session ='1', run = '01')
        anat_bids_path = BIDSPath(root=bids_dir, subject=subject, datatype='anat',
                                  extension='.nii', acquisition = 'MPRAGE', suffix = 'T1w', session = '1')
        '''if os.path.exists(anat_bids_path) == False:
            anat_bids_path = BIDSPath(root=bids_dir, subject=subject, datatype='anat',
                                      extension='.nii', acquisition = 'FSPGR', suffix = 'T1w', session = '1', recording='SCIC')
        else:
            pass'''
        deriv_path = bids_path.copy().update(root=output_dir, check=False)
        deriv_path.directory.mkdir(exist_ok=True, parents=True)
    
        raw_fname = bids_path.copy().update(run = '01', session = '1')
        bem_fname = deriv_path.copy().update(suffix='bem', extension='.fif')
        fwd_fname = deriv_path.copy().update(suffix='fwd', extension='.fif')
        src_fname = deriv_path.copy().update(suffix='src', extension='.fif')
        trans_fname = deriv_path.copy().update(suffix='trans',extension='.fif')
        raw = mne.io.read_raw_ctf(raw_fname.fpath, system_clock = 'ignore', clean_names =True)
        
        subjects_dir = mne_bids.read.get_subjects_dir()
        fs_subject = 'sub-'+bids_path.subject
        if not bem_fname.fpath.exists():
            mne.bem.make_watershed_bem(fs_subject, subjects_dir=subjects_dir, overwrite=True)
            bem = mne.make_bem_model(fs_subject, subjects_dir=f'{subjects_dir}', 
                                     conductivity=[0.3])
            bem_sol = mne.make_bem_solution(bem)
            
            mne.write_bem_solution(bem_fname, bem_sol, overwrite=True)
        else:
            bem_sol = mne.read_bem_solution(bem_fname)
            
        if not src_fname.fpath.exists():
            src = mne.setup_source_space(fs_subject, spacing='oct6', add_dist='patch',
                                 subjects_dir=subjects_dir)
            src.save(src_fname.fpath, overwrite=True)
        else:
            src = mne.read_source_spaces(src_fname.fpath)
    
        if not trans_fname.fpath.exists():
            trans = mne_bids.read.get_head_mri_trans(bids_path, extra_params=dict(system_clock='ignore'),
                                                t1_bids_path=anat_bids_path, fs_subject=fs_subject, 
                                                fs_subjects_dir=subjects_dir)
            mne.write_trans(trans_fname.fpath, trans, overwrite=True)
        else:
            trans = mne.read_trans(trans_fname.fpath)
        #if fwd_fname.fpath.exists():
          #  fwd = mne.read_forward_solution(fwd_fname)
        #else:
        fwd = mne.make_forward_solution(raw.info, trans, src, bem_sol, eeg=False, 
                                        n_jobs=n_jobs)
        mne.write_forward_solution(fwd_fname.fpath, fwd, overwrite=True)
        return {'fwd': fwd, 'trans': trans, 'src': src}


stc_ratios=[]
for file in os.listdir(topdir.joinpath('beamforming', 'epoched', 'time_series' )):
    os.chdir(topdir.joinpath('beamforming', 'epoched', 'time_series' )) 
    with open(file, "rb") as f:
         stc_ratio=pickle.load(f)
    os.chdir(topdir) 
    #stc_morphed = fsaverage_morph(stc_ratio)
    
    
    #stc_ratios.append(stc_morphed)
    stc_ratios.append(stc_ratio)
#stc_ratios is a list of subject time series


average_ts = average_stcs(stc_ratios)

brain = average_ts.plot()
