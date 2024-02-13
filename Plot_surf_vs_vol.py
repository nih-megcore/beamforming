#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 17:25:43 2024

@author: jstout
"""

import glob
import nibabel as nib
import mne, mne_bids
import os, os.path as op
from nih2mne.utilities.bids_helpers import get_mri_dict


nii_files=glob.glob('*.nii')
tmp=False
for i in nii_files:
    if tmp is False:
        tmp=nib.load(i).get_fdata()
    else:
        tmp+=nib.load(i).get_fdata()

mr=nib.load(i)

nifti = nib.Nifti1Image(tmp/len(nii_files), affine=mr.affine)
nifti.to_filename('/tmp/VOL/AVE.nii')



#%%
PROJECT = 'nihmeg'

bids_root = '/data/ML_MEG/NIH_hvmeg'
deriv_root = op.join(bids_root, 'derivatives')
subjects_dir = op.join(deriv_root, 'freesurfer','subjects')
project_root = op.join(deriv_root, PROJECT)
results_root = op.join(project_root, 'RESULTS')
if not op.exists(results_root): os.mkdir(results_root)


# subject='ON39099'
deriv_path = mne_bids.BIDSPath(root=project_root,
                               # subject=subject,
                              extension='.fif', 
                              check=False, 
                              session='01',
                              run='01', 
                              datatype='meg', 
                              task='sternberg')

#%% Surface based results

def visualizer(deriv_path=None, proc='surf'):
    subject=deriv_path.subject
    # lims = [0.3, 0.45, 0.6] #color gradient
    tmp = deriv_path.copy().update(description='Enc6o4ratioFS', suffix='meg',extension='.stc',
                                   processing=proc)
    stc_path = str(tmp).replace('_meg','_meg-lh')
    stc = mne.read_source_estimate(stc_path)
    clims = dict(pos_lims=[0.05, 0.07, 0.1], kind='value')
    
    frame = stc.plot('fsaverage', subjects_dir=subjects_dir, clim=clims) 
    frame.save_image(op.join(results_root, f'{subject}_surf_lh.png'))
    frame.close()
    
    frame = stc.plot('fsaverage', subjects_dir=subjects_dir, hemi='rh', clim=clims) 
    frame.save_image(op.join(results_root, f'{subject}_surf_rh.png')) 
    frame.close()
    
for subject in glob.glob(op.join(project_root, 'sub-*')):
    subject = op.basename(subject)[4:]
    print(subject)
    deriv_path.update(subject=subject)
    try:
        visualizer(deriv_path)
    except:
        print(f'Error with: {subject}')


#%% Volume Based results projected to surface

import nilearn
from nilearn import datasets
import numpy as np
from nilearn import surface
from nilearn import plotting
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt

fsaverage = datasets.fetch_surf_fsaverage()


def projection_plot(threshold=0.05, nii_fname=None):
    curv_right = surface.load_surf_data(fsaverage.curv_right)
    curv_right_sign = np.sign(curv_right)

    curv_left = surface.load_surf_data(fsaverage.curv_left)
    curv_left_sign = np.sign(curv_left)
    
    texture_rh = surface.vol_to_surf(nii_fname, fsaverage.pial_right)
    texture_lh = surface.vol_to_surf(nii_fname, fsaverage.pial_left)
    
    fig, ax = subplots(1,2, subplot_kw={'projection': '3d'})
    
    ## Lateral
    plotting.plot_surf_stat_map(
        fsaverage.infl_left, texture_lh, hemi='left',
        title='Surface left hemisphere', colorbar=True,
        threshold=threshold, bg_map=curv_left_sign, axes=ax[0], 
        symmetric_cbar=True
    )
    
    plotting.plot_surf_stat_map(
        fsaverage.infl_right, texture_rh, hemi='right',
        title='Surface right hemisphere', colorbar=True,
        threshold=threshold, bg_map=curv_right_sign, axes=ax[1],
        symmetric_cbar=True
    )
    
    out_fname = op.join(results_root, f'{subject}_volproj.png')
    fig.savefig(out_fname)
    plt.close(fig)
    


for subject in glob.glob(op.join(project_root, 'sub-*')):
    subject = op.basename(subject)[4:]
    print(subject)
    try:
        nii_fname = glob.glob(op.join(project_root, 'sub-'+subject, 'ses-01', 'meg', '*Enc6o4RatioFS_meg.nii'))[0]
        projection_plot(threshold=0.05, nii_fname=nii_fname)
    except:
        print(f'Error with: {subject}')


# def visualizer_grid(fmin, fmax):
#     lims = [0.3, 0.45, 0.6] #color gradient
#     stcs = all_subj_beamformer(fmin, fmax)[1]
#     src = mne.setup_source_space(fsaverage, spacing='oct6', add_dist='patch')
#     for i in stcs:
#         kwargs = dict(
#             src=src_fs,
#             subjects_dir=subj_dir,
#             subject='fsaverage',
#             initial_time=0,
#             verbose=True,
#             hemi = 'split'
#         )
#     stc.plot(clim=dict(kind="value", lims=lims), **kwargs)  #figure out how to get grid of plots
