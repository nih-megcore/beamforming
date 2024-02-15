#!/usr/bin/env python3

"""
Created on Wed Feb 14 11:58:43 2024

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
import nih2mne
from nih2mne.utilities.bids_helpers import get_mri_dict
n_jobs = 10 

# =============================================================================
# Prep
# Run megcore_mriprep.py before running this code
# =============================================================================


topdir = pathlib.PurePath('/data/ML_MEG') # topdir = '/data/ML_MEG'
resultsdir = pathlib.PurePath(op.join(topdir, 'results_s_final'))
os.chdir(topdir)
bids_root = '/data/ML_MEG/NIH_hvmeg/'
deriv_root = op.join(bids_root, 'derivatives')
project_root = op.join(deriv_root, 'nihmeg')


subj_dir = pathlib.PurePath(op.join(topdir, 'NIH_hvmeg/derivatives/freesurfer/subjects'))

#Setup for fs average warping
subjects_dir = subj_dir
fetch_fsaverage(subjects_dir, verbose=False)  # ensure fsaverage src exists


fsaverage = '/data/ML_MEG/NIH_hvmeg/derivatives/freesurfer/subjects/fsaverage'


def fsaverage_morph(stc, subj_src=None, src_type='surf'):
    stc = copy.deepcopy(stc)
    #Setup for fs average warping
    fetch_fsaverage(subjects_dir, verbose=False)  # ensure fsaverage src exists
    if src_type=='surf':
        fname_fs_src = subjects_dir.joinpath('fsaverage','bem','fsaverage-ico-5-src.fif')
        src_fs = mne.read_source_spaces(fname_fs_src)
    else:
        fname_fs_src = subjects_dir.joinpath('fsaverage','bem', 'fsaverage-vol-5-src.fif')
        src_fs = mne.read_source_spaces(fname_fs_src)
    src_fs = mne.read_source_spaces(fname_fs_src)
    morph = mne.compute_source_morph(
        subj_src,
        subject_from=stc.subject,
        src_to=src_fs,
        subjects_dir=subjects_dir,
        niter_sdr=[5, 5, 4],
        niter_affine=[100, 100, 20],
        zooms=3,  
        verbose=True,
    )
    stc = morph.apply(stc)
    return stc


#%% 
def full_process(subjid, fmin=None, fmax=None, vol_src=False):
    if vol_src == True: 
        src_type='vol' 
    else:
        src_type='surf'
    
    filename = glob.glob(op.join(bids_root, 'sub-'+subjid, '*','meg','*sternberg*.ds'))[0]
    noise_fname = glob.glob(op.join(bids_root, 'sub-'+subjid, '*','meg','*noise*.ds'))[0]

    raw=mne.io.read_raw_ctf(filename, preload=True, clean_names=True, system_clock='ignore') #f'{topdir}/data/movie/{filename}')
    raw.resample(150, n_jobs=n_jobs)
    raw.notch_filter([60], n_jobs=n_jobs)
    raw.filter(fmin, fmax, n_jobs=n_jobs)
    
    raw_eroom = mne.io.read_raw_ctf(noise_fname, preload=True, clean_names=True, system_clock='ignore')
    raw_eroom.resample(150, n_jobs=n_jobs)
    raw_eroom.notch_filter([60], n_jobs=n_jobs)
    raw_eroom.filter(fmin, fmax)
    
    mri_dict = get_mri_dict(subjid, task='sternberg', project='nihmeg', bids_root=bids_root)
    if vol_src==False:
        fwd = mri_dict['fwd'].load()
        src = fwd['src']
    else:
        fwd = mri_dict['volfwd'].load()
        src = fwd['src']
        
    if src_type=='surf':
        fname_fs_src = subjects_dir.joinpath('fsaverage','bem','fsaverage-ico-5-src.fif')
        src_fs = mne.read_source_spaces(fname_fs_src)
    else:
        fname_fs_src = subjects_dir.joinpath('fsaverage','bem', 'fsaverage-vol-5-src.fif')
        src_fs = mne.read_source_spaces(fname_fs_src)
        
    evts, evtsid = mne.events_from_annotations(raw)
    evtsid_4_6 = {i:j for i,j in evtsid.items() if i in ['encode4','encode6']}
    epo = mne.Epochs(raw, evts, event_id=evtsid_4_6, preload=True, tmin=0, tmax=2, baseline=None) # define epo
    epo4 = epo['encode4']
    epo6 = epo['encode6']
    
 
    epo_noise = mne.make_fixed_length_epochs(raw_eroom, duration=2, preload=True)
    
    # ica = mne.preprocessing.read_ica(ica_fname) ##FIX THIS
    # ica.apply(epo_noise)
    
    raw_chs = epo.ch_names
    er_chs = epo_noise.ch_names
    final_chs=list(set(raw_chs).intersection(set(er_chs)))
    epo.pick(final_chs)
    epo_noise.pick(final_chs)
    
    
    #Make list of covariances for each 1s trial
    common_cov = mne.compute_covariance(epo, tmin=0, tmax=2, 
                                        method='shrunk', cv=5, n_jobs=n_jobs)   
    
    noise_cov = mne.compute_covariance(epo_noise, n_jobs=n_jobs, method='shrunk',cv=5)
    
    noise_rank = mne.compute_rank(epo_noise)
    epo_rank = mne.compute_rank(epo)
    if 'mag' in epo_rank:
        if epo_rank['mag'] < noise_rank['mag']:
            noise_rank['mag']=epo_rank['mag']
  
    
    filters = make_lcmv(epo.info, fwd, common_cov, noise_cov=noise_cov, 
                        reg=0.05, rank=epo_rank, pick_ori='max-power')  

    if vol_src == False:   
        #Adjust filters to align with anatomy
        src = fwd['src']
        src_anat_ori = np.vstack([i['nn'][i['vertno']] for i in src])
        ori_flip = np.sign(np.sum(filters['max_power_ori'] * src_anat_ori, axis=1))
        filters['weights'] *= ori_flip[:,np.newaxis]
        filters['max_power_ori'] *= ori_flip[:,np.newaxis]

    stc_noise = apply_lcmv_cov(noise_cov, filters)
    
    #APPLY FILTER TO MAKE STC LIST OF EPOCHS
    stcs4 = apply_lcmv_epochs(epochs=epo4, filters=filters, return_generator=False)  
    stcs6 = apply_lcmv_epochs(epochs=epo6, filters=filters, return_generator=False) 
    
    #Normalize with noise
    for stc in stcs4+stcs6:
        stc._data = stc._data**2
        stc._data /= stc_noise._data
    
    
    stc4 = np.mean(stcs4).mean() 
    stc6 = np.mean(stcs6).mean()
    stc_ratio = copy.deepcopy(stc4)
    stc_ratio._data = np.log(stc6._data / stc4._data)
    
    bids_path = mne_bids.get_bids_path_from_fname(filename)
    deriv_path = bids_path.copy().update(root=project_root, check=False, 
                                         extension=None,datatype='meg', 
                                         processing=src_type)
    
    deriv_path.fpath.parent.mkdir(parents=True, exist_ok=True)
    stc4_fname = deriv_path.copy().update(description='Enc4')
    stc6_fname = deriv_path.copy().update(description='Enc6')
    stc_ratio_fname = deriv_path.copy().update(description='Enc6o4Ratio')
    
    stc4.save(stc4_fname.fpath)
    stc6.save(stc6_fname.fpath)

    stc_ratio_fs=fsaverage_morph(stc_ratio, subj_src=src, src_type=src_type)
    
    stc_ratio_fs_fname = deriv_path.copy().update(description='Enc6o4RatioFS')
    if src_type == 'vol':
        stc_ratio.save_as_volume(stc_ratio_fname.update(extension='.nii'), src)
        stc_ratio_fs_fname.update(extension='.nii')
        stc_ratio_fs.save_as_volume(stc_ratio_fs_fname.fpath, src_fs)
    else:
        stc_ratio.save(stc_ratio_fname.fpath)
        stc_ratio_fs.save(stc_ratio_fs_fname.fpath) 
    

# subjects=['sub-'ON39099'',
#       'sub-ON41090',
#       'sub-ON42107',
#       'sub-ON43016',
#       'sub-ON43585',
#       'sub-ON47254',
#       'sub-ON48925',
#       'sub-ON49080',
#       'sub-ON52083',
#       'sub-ON52662',
#       'sub-ON54268']
# subjects=[i[4:] for i in subjects]

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Process data based on subject, min/max freq, and volume/surf")
    parser.add_argument("-subjid", type=int, help="ID of the subject")
    parser.add_argument("-fmin", type=float, help="Minimum frequency")
    parser.add_argument("-fmax", type=float, help="Maximum frequency")
    parser.add_argument("-vol_src", default=False, action = 'store_true', help="True means volume, False means surface")
    
    args = parser.parse_args()
    
    full_process(args.subjid, args.fmin, args.fmax, args.vol_src)
