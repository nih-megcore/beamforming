# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:15:43 2024

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

n_jobs = 20 #CHANGE

topdir = pathlib.PurePath('/data/ML_MEG') # topdir = '/data/ML_MEG'
resultsdir = pathlib.PurePath(op.join(topdir, 'results_s_final'))
os.chdir(topdir)

#stcdir = op.join(topdir, resultsdir, f'f_{fmin}_{fmax}')
#lh_stc = mne.read_source_estimate('ON25658_stc.h5-lh.stc')
#rh_stc = mne.read_source_estimate('ON25658_stc.h5-rh.stc')

subj_dir = pathlib.PurePath(op.join(topdir, 'NIH_hvmeg/derivatives/freesurfer/subjects'))

#Setup for fs average warping
subjects_dir = subj_dir
fetch_fsaverage(subjects_dir, verbose=False)  # ensure fsaverage src exists
fname_fs_src = subjects_dir.joinpath('fsaverage','bem','fsaverage-vol-5-src.fif')
src_fs = mne.read_source_spaces(fname_fs_src)

fsaverage = '/data/ML_MEG/NIH_hvmeg/derivatives/freesurfer/subjects/fsaverage'

'''
morphed_stcs=[]
os.chdir('X:\\beamforming\\epoched\\time_series_vol_morphed')
for i in os.listdir('X:\\beamforming\\epoched\\time_series_vol_morphed'):
    with open(i, "rb") as f:
        stc=pickle.load(f)
    morphed_stcs.append(stc)
'''

def visualizer(fmin, fmax):
    lims = [0.3, 0.45, 0.6] #color gradient
    stc = all_subj_beamformer(fmin, fmax)[0]
    src = mne.setup_source_space(fsaverage, spacing='oct6', add_dist='patch')
    kwargs = dict(
        src=src_fs,
        subjects_dir=subj_dir,
        subject='fsaverage',
        initial_time=0,
        verbose=True,
        hemi = 'split'
    )
    stc.plot(clim=dict(kind="value", lims=lims), **kwargs)  
    

def visualizer_grid(fmin, fmax):
    lims = [0.3, 0.45, 0.6] #color gradient
    stcs = all_subj_beamformer(fmin, fmax)[1]
    src = mne.setup_source_space(fsaverage, spacing='oct6', add_dist='patch')
    for i in stcs:
        kwargs = dict(
            src=src_fs,
            subjects_dir=subj_dir,
            subject='fsaverage',
            initial_time=0,
            verbose=True,
            hemi = 'split'
        )
    stc.plot(clim=dict(kind="value", lims=lims), **kwargs)  #figure out how to get grid of plots

def all_subj_beamformer(fmin, fmax):
    all_stcs_ratios = []
    directory = pathlib.PurePath(op.join(topdir, 'NIH_hvmeg/derivatives/ica'))
    errors = []
    for i in os.listdir(directory):
        filename = pathlib.PurePath(op.join(directory, i, 'ica_clean.fif'))
        print(filename)
        try:
            stc_ratio_fs=full_process(filename, fmin, fmax)
            all_stcs_ratios.append(stc_ratio_fs)
        except BaseException as e:
            print(f'Error {str(e)} {i}')
            errors.append(f'Error {str(e)} {i}')
    final_stc_ratio = average_stcs(all_stcs_ratios)
    print(errors)
    return final_stc_ratio, all_stcs_ratios
    
    '''subj_pd = pd.read_csv('X:\\good_subjects_stern_ica.csv')
    good_subjs = subj_pd.iloc[:,1].tolist()
    for i in good_subjs:
        #filename = pathlib.PurePath(op.join(directory, i, 'ica_clean.fif'))
        filename = pathlib.PurePath(i) 
        print(filename)
        try:
            stc_ratio_fs=full_process(filename, fmin, fmax)
            all_stcs_ratios.append(stc_ratio_fs)
        except BaseException as e:
            print(f'Error {str(e)} {i}')
            errors.append(f'Error {str(e)} {i}')
    final_stc_ratio = average_stcs(all_stcs_ratios)
    print(errors)
    return final_stc_ratio'''


'''
morphed_stcs = []
for stc in all_stcs_ratios:
    morph_stc = fsaverage_morph(stc)
    morphed_stcs.append(morph_stc)

ave_morph_stc = average_stcs(morphed_stcs)
'''
def fsaverage_morph(stc):
    
    #Setup for fs average warping
    subjects_dir = subj_dir
    fetch_fsaverage(subjects_dir, verbose=False)  # ensure fsaverage src exists
    fname_fs_src = subjects_dir.joinpath('fsaverage','bem', 'fsaverage-vol-5-src.fif')
    src_fs = mne.read_source_spaces(fname_fs_src)
    src = get_mri(stc.subject[4:])['src']
    morph = mne.compute_source_morph(
        src,
        subject_from='sub-'+stc.subject[4:],
        src_to=src_fs,
        subjects_dir=subj_dir,
        niter_sdr=[5, 5, 3],
        niter_affine=[100, 100, 10],
        zooms= 'auto',  # just for speed
        verbose=True,
    )

    stc = morph.apply(stc)
    return stc

def get_mri(subject):
    if subject[0:2] == 'ON':
        #subject = 'ON02747'    
        bids_dir = str(topdir.joinpath('NIH_hvmeg'))
        output_dirname = 'sternberg_vol'      ###RESET NAME!!!!
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
        src_surf_fname = deriv_path.copy().update(suffix='src_surf', extension='.fif')
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
            
        if not src_surf_fname.fpath.exists():
            src_surf = mne.setup_source_space(fs_subject, spacing='oct6', add_dist='patch', subjects_dir=subjects_dir)
            src_surf.save(src_surf_fname.fpath, overwrite=True)
        else:
            src_surf = mne.read_source_spaces(src_surf_fname.fpath)
        '''
        #Use src_surf to create points for src_vol
        pos_src = {}
        pos_src['rr'] = np.concatenate((src_surf[0]['rr'], src_surf[1]['rr']))
        pos_src['nn'] = np.concatenate((src_surf[0]['nn'], src_surf[1]['nn']))
        src = mne.setup_volume_source_space(subject=fs_subject, pos=pos_src, bem=bem_fname,
                                    mindist=2.5, exclude=10.0, subjects_dir=subjects_dir, 
                                    add_interpolator=True, verbose=True)
        '''
        if not src_fname.fpath.exists():
            src = mne.setup_volume_source_space(subject=fs_subject, pos=5, bem=bem_fname,
                                        mindist=2.5, exclude=10.0, subjects_dir=subjects_dir, 
                                        add_interpolator=True, verbose=True)
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
        bids_dir = str(topdir.joinpath('NIH_hvmeg'))
        output_dirname = 'sternberg_vol'      ###RESET NAME!!!!
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
        fwd_fname = deriv_path.copy().update(suffix='old_fwd', extension='.fif')
        src_fname = deriv_path.copy().update(suffix='old_src', extension='.fif')
        src_surf_fname = deriv_path.copy().update(suffix='src_surf', extension='.fif')
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
            
        if not src_surf_fname.fpath.exists():
            src_surf = src = mne.setup_source_space(fs_subject, spacing='oct6', add_dist='patch', subjects_dir=subjects_dir)
            src.save(src_surf_fname.fpath, overwrite=True)
        else:
            src_surf = mne.read_source_spaces(src_surf_fname.fpath)
        '''
        #Use src_surf to create points for src_vol
        pos_src = {}
        pos_src['rr'] = np.concatenate((src_surf[0]['rr'], src_surf[1]['rr']))
        pos_src['nn'] = np.concatenate((src_surf[0]['nn'], src_surf[1]['nn']))
        pos_src['tris'] = np.concatenate((src_surf[0]['tris'], src_surf[1]['tris']))
        src = mne.setup_volume_source_space(subject=fs_subject,
                                    mindist=2.5, exclude=10.0, subjects_dir=subjects_dir, 
                                    add_interpolator=True, verbose=True, surface = pos_src)
        src.save(src_fname.fpath, overwrite=True)
        '''
        
        if not src_fname.fpath.exists():
            src = mne.setup_volume_source_space(subject=fs_subject, pos=5, bem=bem_fname,
                                        mindist=2.5, exclude=10.0, subjects_dir=subjects_dir, 
                                        add_interpolator=True, verbose=True)
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


def full_process(filename, fmin, fmax):
    subjid = filename.parents[0].name.split('-')[1][:-4]
    
    if subjid[0:2] == 'ON':
        fwd_dir = topdir.joinpath('NIH_hvmeg','derivatives',f'sternberg_res_{fmin}_{fmax}_final',f'sub-{subjid}','ses-01','meg')
        fwd_fname = fwd_dir.joinpath(f'sub-{subjid}_ses-01_task-sternberg_run-01_fwd.fif')
        subject_fs = subjid+'_fs'
        ica_fname = topdir.joinpath('NIH_hvmeg','derivatives','ica',f'sub-{subjid}_ses-01_task-sternberg_run-01_meg',f'sub-{subjid}_ses-01_task-sternberg_run-01_meg_0-ica.fif')
        noise_fname = topdir.joinpath('NIH_hvmeg',f'sub-{subjid}','ses-01','meg',f'sub-{subjid}_ses-01_task-noise_run-01_meg.ds')
    else:
        fwd_dir = topdir.joinpath('NIH_hvmeg','derivatives',f'sternberg_res_{fmin}_{fmax}_final',f'sub-{subjid}','ses-1','meg')
        fwd_fname = fwd_dir.joinpath(f'sub-{subjid}_ses-1_task-sternberg_run-01_fwd.fif')
        subject_fs = subjid+'_fs'
        ica_fname = topdir.joinpath('NIH_hvmeg','derivatives','ica',f'sub-{subjid}_ses-1_task-sternberg_run-01_meg',f'sub-{subjid}_ses-1_task-sternberg_run-01_meg_0-ica.fif')
        noise_fname = topdir.joinpath('NIH_hvmeg',f'sub-{subjid}','ses-1','meg',f'sub-{subjid}_ses-1_task-noise_run-01_meg.ds')
    
    
    '''
    #subjid = filename.split('/')[2][8:15]  #subjid = op.basename(filename).split('_')[0][4:] #
    subjid = filename.split('/')[2][4:11] #6 for /data/ML_MEG but 3 for X drive
    fwd_dir = op.join(topdir, f'NIH_hvmeg/derivatives/sternberg_res_{fmin}_{fmax}_final/sub-{subjid}/ses-01/meg')
    fwd_fname= op.join(fwd_dir, f'sub-{subjid}_ses-01_task-sternberg_run-01_fwd.fif')     
    subject_fs = subjid+'_fs'
    ica_fname = f'X:\\NIH_hvmeg/derivatives/ica/sub-{subjid}_ses-01_task-sternberg_run-01_meg/sub-{subjid}_ses-01_task-sternberg_run-01_meg_0-ica.fif'
    noise_fname = f'X:\\NIH_hvmeg/sub-{subjid}/ses-01/meg/sub-{subjid}_ses-01_task-noise_run-01_meg.ds'
    '''

    raw=mne.io.read_raw_fif(filename) #f'{topdir}/data/movie/{filename}')

    raw = raw.load_data() #mne.io.read_raw_ctf(filename, preload=True)
    
    raw.resample(150, n_jobs=n_jobs)
    
    raw.notch_filter([60], n_jobs=n_jobs)
    raw.filter(fmin, fmax) #15.0,25)
    anat_dict =  get_mri(subjid)
    fwd = anat_dict['fwd']
    
    os.chdir('/data/ML_MEG/beamforming/epoched_2/fwd_vol')
    with open(f"{subjid}_fwd.pkl", "wb") as fp:
        pickle.dump(fwd, fp)  
    os.chdir(topdir)
    
    evts, evtsid = mne.events_from_annotations(raw)
    e = [x for x in evts if x[2] == 1 or x[2] == 2]#  x[2] == ?]
    e4 = [x for x in evts if x[2] == 1]
    e6 = [x for x in evts if x[2] == 2]
    epo = mne.Epochs(raw, e, preload=True, tmin=0, tmax=2, baseline=None) # define epo
    epo4 = mne.Epochs(raw, e4, preload=True, tmin=0, tmax=2, baseline=None)
    epo6 = mne.Epochs(raw, e6, preload=True, tmin=0, tmax=2, baseline=None)
    
    # =============================================================================
    #     Caculate covariance and beamformer off of cleaned trials
    # =============================================================================
    full_cov = mne.compute_covariance(epo)
    raw_eroom = mne.io.read_raw_ctf(noise_fname, preload=True, clean_names=True, system_clock='ignore') ########################################
    
    epo_noise = mne.Epochs(raw_eroom, e, preload=True)
    
    ica = mne.preprocessing.read_ica(ica_fname) ##FIX THIS
    ica.apply(epo_noise)
    
    raw_chs = epo.ch_names
    er_chs = epo_noise.ch_names
    final_chs=list(set(raw_chs).intersection(set(er_chs)))
    epo.pick(final_chs)
    epo_noise.pick(final_chs)
    
    
    #Make list of covariances for each 1s trial
    common_cov = mne.compute_covariance(epo, tmin=0, tmax=2, 
                                        method='shrunk', cv=5, n_jobs=n_jobs)   
    
    #noise_cov = mne.compute_covariance(epo, tmin = -0.4, tmax = 0, method = 'shrunk', cv =5, n_jobs=n_jobs)
    noise_cov = mne.compute_covariance(epo_noise)
    
    noise_rank = mne.compute_rank(epo_noise)
    epo_rank = mne.compute_rank(epo)
    if 'mag' in epo_rank:
        if epo_rank['mag'] < noise_rank['mag']:
            noise_rank['mag']=epo_rank['mag']
  
    #evoked = epo.average()
    
    filters = make_lcmv(epo.info, fwd, full_cov, noise_cov=noise_cov, 
                        reg=0.05, rank=epo_rank, pick_ori='max-power')  

   
    #Adjust filters to align with anatomy
    src = fwd['src']
    src_anat_ori = np.vstack([i['nn'][i['vertno']] for i in src])
    ori_flip = np.sign(np.sum(filters['max_power_ori'] * src_anat_ori, axis=1))
    filters['weights'] *= ori_flip[:,np.newaxis]
    filters['max_power_ori'] *= ori_flip[:,np.newaxis]
    
    #APPLY FILTER TO MAKE STC LIST OF EPOCHS
    stcs4 = apply_lcmv_epochs(epochs=epo4, filters=filters, return_generator=False)  
    stcs6 = apply_lcmv_epochs(epochs=epo6, filters=filters, return_generator=False)  

# =============================================================================
#     #Save out the stcs6 and the stcs4 to mapped drive
#     os.chdir('X:\\beamforming\epoched')
#     with open(f"{subjid}_stcs6.pkl", "wb") as fp:
#         pickle.dump(stcs6, fp)
#     with open(f"{subjid}_stcs4.pkl", "wb") as fp:
#         pickle.dump(stcs4, fp)     
#     os.chdir(topdir)
# =============================================================================
    
# =============================================================================
# 
#     #Open the stcs6 and stcs4
#     os.chdir('X:\\beamforming\epoched')
#     with open(f"{subjid}_stcs6.pkl", "rb") as f:
#         stcs6=pickle.load(f)
#     with open(f"{subjid}_stcs4.pkl", "rb") as f:
#         stcs4=pickle.load(f)
# =============================================================================
    
    #SQUARE EACH VALUE IN STC
    stcs4_sq = [i**2 for i in stcs4]
    stcs6_sq = [i**2 for i in stcs6]
    #stc_noise = apply_lcmv_cov(noise_cov, filters)
    #stcs_sq_ave4.data /= stc_noise.data#[np.newaxis,:]

    #AVERAGE ACROSS EPOCHS
    stc4_av = average_stcs(stcs4_sq)
    stc6_av = average_stcs(stcs6_sq)
    
    #AVERAGE ACROSS TIME
    stc4_fin = stc4_av.mean()
    stc6_fin = stc6_av.mean()
    
    #SAVE OUT STC 4 and STC6
    os.chdir('/data/ML_MEG/beamforming/epoched_2/stc_4_vol')
    with open(f"{subjid}_stc4.pkl", "wb") as fp:
        pickle.dump(stc4_fin, fp)  
    os.chdir(topdir)
    
    os.chdir('/data/ML_MEG/beamforming/epoched_2/stc_6_vol')
    with open(f"{subjid}_stc6.pkl", "wb") as fp:
        pickle.dump(stc6_fin, fp)  
    os.chdir(topdir)
    
    
    #FIND THE LOG RATIO
    stc_ratio = stc4_fin.copy()
    stc_ratio.data = np.log(stc6_fin.data/stc4_fin.data)
    #stc_noise = apply_lcmv_cov(noise_cov, filters)
    
    #Save out the fs stc to mapped drive
    #os.chdir('X:\\beamforming\\epoched\\time_series_vol')
    os.chdir('/data/ML_MEG/beamforming/epoched2/time_series_vol')
    with open(f"{subjid}_unmorphed_ratio.pkl", "wb") as fp:
        pickle.dump(stc_ratio, fp)  
    os.chdir(topdir)
    
    #MORPHING TO STANDARD SPACE
    #stc_ratio_fs=fsaverage_morph(stc_ratio)

# =============================================================================
#     #Save out the fs stc to mapped drive
#     os.chdir('X:\\beamforming\\epoched\\time_series')
#     with open(f"{subjid}_morph_ratio.pkl", "wb") as fp:
#         pickle.dump(stc_ratio_fs, fp)  
#     os.chdir(topdir)
# =============================================================================

    return stc_ratio