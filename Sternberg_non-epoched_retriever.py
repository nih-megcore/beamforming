# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:18:33 2024

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

fsaverage = 'X:/NIH_hvmeg/derivatives/freesurfer/subjects/fsaverage'

# =============================================================================
# 1) Read in all stcs
# 2) For each stc, make list of stcs that is a list of epochs and then avaerage across all epochs
#     -> result is list of stcs wher each one is average across all epochs and is a different subject
# 3) morph stcs to fs average 
# 4) avergae stcs across al subjects
# 5) plot the stc
# =============================================================================

def visualizer(fmin, fmax):
    lims = [0.3, 0.45, 0.6] #color gradient
    stc = all_subj_retriever(fmin, fmax)
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


def all_subj_retriever(fmin, fmax):
    all_stcs_ratios = []
    #directory = pathlib.Path(op.join(topdir, 'results_s_final', f'f_{fmin}_{fmax}')).glob('*lh.stc')
    directory = pathlib.Path(topdir.joinpath('results_s_final', f'f_{fmin}_{fmax}')).glob('*lh.stc')
    errors = []
    means = {}
    mins = {}
    maxs = {}
    stdevs = {}
    for file in directory:
        try:
            #stc_ratio_fs=stc_retriever(file, fmin, fmax)['stc_image']
            #all_stcs_ratios.append(stc_ratio_fs)
            stc_dict = stc_retriever(file, fmin, fmax)
            means[stc_dict['subjid']] = stc_dict['meann']
            mins[stc_dict['subjid']] = stc_dict['minn']
            maxs[stc_dict['subjid']] = stc_dict['maxx']
            stdevs[stc_dict['subjid']] = stc_dict['stdevv']
        except BaseException as e:
            print(f'Error {str(e)} {file}')
            errors.append(f'Error {str(e)} {file}')
    #final_stc_ratio = average_stcs(all_stcs_ratios)
    print(errors)
    #return final_stc_ratio


def stc_retriever(file, fmin, fmax):
    evt_dir = topdir.joinpath('NIH_hvmeg', 'derivatives', 'ica')
    evt_errors = []
    
    print(file)
    stcs4 = []
    stcs6 = []
    evt_errors = []
    subjid = os.path.basename(file).split('_')[0]

    for evt_file in os.listdir(evt_dir):
        if subjid in evt_file:
            full_evt_file = evt_dir.joinpath(evt_file, 'ica_clean.fif')
            try:
                raw=mne.io.read_raw_fif(full_evt_file) 
                raw = raw.load_data()
                raw.resample(150, n_jobs=n_jobs)
                evts, evtsid = mne.events_from_annotations(raw)
                evts_4 = [x[0] for x in evts if x[2] == 1]
                evts_6 = [x[0] for x in evts if x[2] == 2]
            except BaseException as e:
                print(f'Error {str(e)} {i}')
                evt_errors.append(f'Error {str(e)} {i}')
        else:
            pass
    print(f'event errors are: {evt_errors}')
    stc = mne.read_source_estimate(file)
    stc.subject = f'sub-{subjid}'
    for ev4 in evts_4:
        stcs4.append(stc.copy().crop(ev4/150, ev4/150+2)) #Assumes that sampling freq is 150 Hz
    for ev6 in evts_6:
        stcs6.append(stc.copy().crop(ev6/150, ev6/150+2)) #Assumes that sampling freq is 150 Hz

    #SQUARE EACH VALUE IN STC
    stcs4_sq = [i**2 for i in stcs4]
    stcs6_sq = [i**2 for i in stcs6]
    #stc_noise = apply_lcmv_cov(noise_cov, filters)
    #stcs_sq_ave4.data /= stc_noise.data#[np.newaxis,:]

    #AVERAGE ACROSS EPOCHS
    stc4_av = average_stcs(stcs4_sq)
    stc6_av = average_stcs(stcs6_sq)
    
    stc_image = stc6_av.copy()
    stc_image.data = np.log(stc6_av.data/stc4_av.data)
    stc_image.tmin  = 0.0 #CHANGE DEPENDING ON CONTEXT

    meann= np.mean(stc_image.data)
    minn=np.min(stc_image.data)
    maxx=np.max(stc_image.data)
    stdevv=np.std(stc_image.data)
    
    
    #Save out the stc_image to mapped drive
    os.chdir('X:\\beamforming\\no_epochs\\time_series')
    with open(f"{subjid}_stc_ts.pkl", "wb") as fp:
        pickle.dump(stc_image, fp)  
    os.chdir(topdir)
    
    '''
    #Plot out only the negative values
    stc_image.data[stc_image.data > 0] = 0
    stc_image *= -1
    '''
    '''
    #Plotting to make the image
    os.chdir(topdir.joinpath('beamforming', 'no_epochs', 'subject_images'))
    
    brain = stc_image.copy().crop(tmin=0.5,tmax=0.5).plot(hemi='split', views=['lat', 'med'], size=(1600, 1600), colormap='cool', time_viewer=True, clim='auto')
    brain.save_image(f'neg_{subjid}_0.5s.png')
    
    brain.close()
    
    brain = stc_image.copy().crop(tmin=1.5,tmax=1.5).plot(hemi='split', views=['lat', 'med'], size=(1600, 1600), colormap='cool', time_viewer=True, clim='auto')
    brain.save_image(f'neg_{subjid}_1.5s.png')
    
    brain.close()
    
    '''
    '''
    #AVERAGE ACROSS TIME
    stc4_fin = stc4_av.mean()
    stc6_fin = stc6_av.mean()
    
    #FIND THE LOG RATIO
    stc_ratio = stc4_fin.copy()
    stc_ratio.data = np.log(stc6_fin.data/stc4_fin.data)
    stc_ratio.subject = f'sub-{subjid}'
    #stc_noise = apply_lcmv_cov(noise_cov, filters)
    
    #MORPHING TO STANDARD SPACE
    
    stc_ratio_fs=fsaverage_morph(stc_ratio)
    '''   
    #return {'stc_ratio_fs': stc_ratio_fs, 'stc_image': stc_image}    
    return {'meann' : meann, 'minn': minn, 'maxx': maxx, 'stdevv': stdevv, 'subjid': subjid}



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

# =============================================================================
# Comvbines multiple images onto one grid for viewing ease
# =============================================================================

# Import the required modules
os.chdir("X:\\beamforming\\no_epochs\\subject_images")
# Import the required modules
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Define the number of images per row and the spacing between them
columns = 10
space = 20

# Load the images from a list of file names
images2 = os.listdir("X:\\beamforming\\no_epochs\\subject_images")
for file in os.listdir("X:\\beamforming\\no_epochs\\subject_images"):
    if file.startswith("neg") or file.endswith("0.5s.png"):
        images2.remove(file)

images = [Image.open(x) for x in images2]

# Get the maximum width and height of the images
widths, heights = zip(*(i.size for i in images))
max_width = max(widths)
max_height = max(heights)

# Create a new image with a white background
total_width = max_width * columns + space * (columns - 1)
total_height = max_height * ((len(images)) // columns + 1) + space * ((len(images)) // columns)+750
new_im = Image.new("RGB", (total_width, total_height), (255, 255, 255))

# Create a draw object and a font object
draw = ImageDraw.Draw(new_im)
font = ImageFont.truetype("arial.ttf", 45)

# Loop over the images and paste them on the new image
x_offset = 0
y_offset = 0
for i, im in enumerate(images):
    # Center the image horizontally and vertically
    x_center = (max_width - im.width) // 2
    y_center = (max_height - im.height) // 2
    new_im.paste(im, (x_offset + x_center, y_offset + y_center))

    # Draw the text below the image
    text = f"{images2[i]}" # You can change this to any text you want
    text_width, text_height = draw.textsize(text, font)
    text_x = x_offset + (max_width - text_width) // 2
    text_y = y_offset + max_height + space // 2
    draw.text((text_x, text_y), text, (0, 0, 0), font)

    # Update the offsets
    x_offset += max_width + space
    if (i + 1) % columns == 0:
        x_offset = 0
        y_offset += max_height + space + text_height + space

# Save the new image
new_im.save("pos_images_1.5s.jpg")



