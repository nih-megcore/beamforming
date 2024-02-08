#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:56:50 2024

@author: bansals3
"""



import pathlib
import os, os.path as op
import mne
from mne.datasets import sample
from mne.datasets import fetch_fsaverage
import matplotlib.pyplot as plt


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

bids_dir = 'X:\\NIH_hvmeg'
output_dirname = 'sternberg_vol'      ###RESET NAME!!!!
task_type = 'sternberg'
# =============================================================================
# general variables
# =============================================================================
deriv_dir = op.join(bids_dir, 'derivatives')
output_dir = op.join(deriv_dir, output_dirname)
subjects_dir = op.join(deriv_dir, 'freesurfer', 'subjects')
os.environ['SUBJECTS_DIR']=subjects_dir


    
n_jobs = 20 #CHANGE
topdir = pathlib.PurePath('X:\\') # topdir = '/data/ML_MEG'

subj_dir = pathlib.PurePath(op.join(topdir, 'NIH_hvmeg/derivatives/freesurfer/subjects'))

#Setup for fs average warping
subjects_dir = subj_dir
subjid = subject = 'EENMOJWI' #'ZCXKMNSF'
subject = subjid
fs_subject = f'sub-{subjid}'


bids_path = BIDSPath(root=bids_dir, subject=subject, datatype='meg',
                     task=task_type, session ='1', run = '01')
anat_bids_path = BIDSPath(root=bids_dir, subject=subject, datatype='anat',
                          extension='.nii', acquisition = 'MPRAGE', suffix = 'T1w', session = '01')


deriv_path = bids_path.copy().update(root=output_dir, check=False)
deriv_path.directory.mkdir(exist_ok=True, parents=True)

raw_fname = bids_path.copy().update(run = '01', session = '1')
bem_fname = deriv_path.copy().update(suffix='bem', extension='.fif')
fwd_fname = deriv_path.copy().update(suffix='fwd', extension='.fif')
src_fname = deriv_path.copy().update(suffix='src', extension='.fif')
src_surf_fname = deriv_path.copy().update(suffix='src_surf', extension='.fif')
trans_fname = deriv_path.copy().update(suffix='trans',extension='.fif')


subj_dir_surf = pathlib.PurePath(op.join(topdir, 'NIH_hvmeg/derivatives/sternberg_surf'))
subj_dir_vol = pathlib.PurePath(op.join(topdir, 'NIH_hvmeg/derivatives/sternberg_vol'))



#Fwd model for surface
fwd_surf_fname = subj_dir_surf / f'sub-{subjid}' / 'ses-1' / 'meg' / f'sub-{subjid}_ses-1_task-sternberg_run-01_fwd.fif'
fwd_surf = mne.read_forward_solution(fwd_surf_fname)

#Fwd model for volume
fwd_vol_fname = subj_dir_vol / f'sub-{subjid}' / 'ses-1' / 'meg' / f'sub-{subjid}_ses-1_task-sternberg_run-01_old_fwd.fif'
fwd_vol = mne.read_forward_solution(fwd_vol_fname)

#Src model for volume
src_vol_fname = subj_dir_vol / f'sub-{subjid}' / 'ses-1' / 'meg' / f'sub-{subjid}_ses-1_task-sternberg_run-01_old_src.fif'
src_vol = mne.read_source_spaces(src_vol_fname)

#Plot fwd model for surface
mag_map_surf = mne.sensitivity_map(fwd_surf, ch_type="mag", mode="fixed")
mag_map_surf.plot()

#Plot fwd model for volume
mag_map_vol = mne.sensitivity_map(fwd_vol, ch_type="mag", mode="fixed")
mag_map_vol.plot_3d(src=fwd_vol['src'], subject=fs_subject)


lims = [0.3, 0.45, 0.6]
kwargs = dict(
    src=src_fs,
    subject='fsaverage',
    subjects_dir=subjects_dir,
    initial_time=0.0,
    verbose=True,
)

brain = mag_map_vol.plot_3d(
    hemi="both",
    size=(600, 600),
    views=["sagittal"],
    # Could do this for a 3-panel figure:
    # view_layout='horizontal', views=['coronal', 'sagittal', 'axial'],
    brain_kwargs=dict(silhouette=True),
    **kwargs,
)

brain_morph = mag_map_vol_morph.plot_3d(
    hemi="both",
    size=(600, 600),
    views=["sagittal"],
    # Could do this for a 3-panel figure:
    # view_layout='horizontal', views=['coronal', 'sagittal', 'axial'],
    brain_kwargs=dict(silhouette=True),
    **kwargs,
)



# plot source space (src_plot.plot())
fwd_vol['src'].plot(trans=trans_fname.fpath)

# extract data from fwd model and plot it 
leadfield_vol = fwd_vol["sol"]["data"]
leadfield_surf = fwd_surf["sol"]["data"]

picks_meg = mne.pick_types(fwd_vol["info"], meg=True, eeg=False)
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Lead field matrix (500 dipoles only)", fontsize=14)

leadfields = [ leadfield_vol, leadfield_surf]
src_mode = ['vol','surf']
ii = 0
for  ax, leadfield in zip(axes, leadfields):
    im = ax.imshow(leadfield[picks_meg, :], origin="lower", aspect="auto", cmap="RdBu_r")
    ax.set_title(src_mode[ii].upper())
    ax.set_xlabel("sources")
    ax.set_ylabel("sensors")
    fig.colorbar(im, ax=ax)
    ii+=1
    
fig_2, ax = plt.subplots()
ax.hist(
    [ mag_map_vol.data.ravel(), mag_map_surf.data.ravel()],
    bins=20,
    label=["Magnetometers_Vol","Magnetometers_Surf"],
    color=["c", "b"],
)
fig_2.legend()
ax.set(title="sensitivity", xlabel="sensitivity", ylabel="count")




### test difference between src and fwd['src'] --> src is in MRI coords; fwd['src'] is in head coords

from mne.transforms import apply_trans

trans = fwd_vol['mri_head_t']['trans']



#coord_surf = fwd_surf['src'][0]['rr']
coord_vol_orig = src_vol[0]['rr'] # this is volume space; with pos argument = int
coord_vol_new = fwd_vol['src'][0]['rr'] # this is also in volume space but with pos argument = dict()

# APPLY transformation matrix to original source space in vol
coord_vol_trans = apply_trans(trans,coord_vol_orig, move=True)




idx = int(75*1e3)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#ax.scatter(coord_surf[:,0],coord_surf[:,1],coord_surf[:,2],marker='d', color='k')
#ax.scatter(coord_vol_orig[:idx,0],coord_vol_orig[:idx,1],coord_vol_orig[:idx,2],marker='.', color='tab:red',alpha=1)
ax.scatter(coord_vol_trans[:idx,0],coord_vol_trans[:idx,1],coord_vol_trans[:idx,2],marker='s', color='k',alpha=0.2)
ax.scatter(coord_vol_new[:idx,0],coord_vol_new[:idx,1],coord_vol_new[:idx,2],marker='*', color='c',alpha=0.2)
ax.view_init(azim=0,elev=0)
plt.show(block=False)




plot_bem_kwargs = dict(
    subject=fs_subject,
    subjects_dir=subj_dir,
    brain_surfaces="white",
    orientation="axial",
    slices=[50, 100, 150, 200],
)


mne.viz.plot_bem(src=src_vol, **plot_bem_kwargs)






#Plot out volume to surface
import nilearn
from nilearn import datasets
import numpy as np
from nilearn import surface
from nilearn import plotting

fname_fs_src = subjects_dir.joinpath('fsaverage','bem', 'fsaverage-vol-5-src.fif')
src_fs = mne.read_source_spaces(fname_fs_src)

def fsaverage_morph(stc):
    
    #Setup for fs average warping
    subjects_dir = subj_dir
    fetch_fsaverage(subjects_dir, verbose=False)  # ensure fsaverage src exists
    fname_fs_src = subjects_dir.joinpath('fsaverage','bem', 'fsaverage-vol-5-src.fif')
    src_fs = mne.read_source_spaces(fname_fs_src)
    fwd = get_mri(stc.subject[4:])['fwd']
    src = fwd['src']
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

mag_map_vol_morph = fsaverage_morph(mag_map_vol)

fsaverage = datasets.fetch_surf_fsaverage()
curv_right = surface.load_surf_data(fsaverage.curv_right)
curv_right_sign = np.sign(curv_right)

curv_left = surface.load_surf_data(fsaverage.curv_left)
curv_left_sign = np.sign(curv_left)

ave_fname = 'sens_map_vol_EEN_morph.nii'
texture_rh = surface.vol_to_surf(ave_fname, fsaverage.pial_right)
texture_lh = surface.vol_to_surf(ave_fname, fsaverage.pial_left)


thresh = 0.1
fig = plotting.plot_surf_stat_map(
    fsaverage.infl_right, texture_rh, hemi='right',
    title='Surface right hemisphere', colorbar=True, cmap='hot',
    threshold=thresh, bg_map=curv_right_sign,
)
fig.show()

fig = plotting.plot_surf_stat_map(
    fsaverage.infl_left, texture_lh, hemi='left',
    title='Surface left hemisphere', colorbar=True, cmap='hot',
    threshold=thresh, bg_map=curv_right_sign,
)
fig.show()