# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:26:47 2024

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
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import nilearn
from nilearn import datasets
import numpy as np
from nilearn import surface
from nilearn import plotting
from matplotlib.pyplot import subplots

n_jobs = 1 #CHANGE
topdir = pathlib.PurePath('X:\\') # topdir = '/data/ML_MEG'
resultsdir = pathlib.PurePath(op.join(topdir, 'results_s_final'))
subj_dir = topdir.joinpath('NIH_hvmeg', 'derivatives', 'freesurfer', 'subjects')

imp_dir = topdir.joinpath('beamforming', 'testing_2')

fname_fs_src = subj_dir.joinpath('fsaverage','bem', 'fsaverage-vol-5-src.fif')
src_fs = mne.read_source_spaces(fname_fs_src)



    
with open('ON02747_unmorphed_ratio.pkl', "rb") as f:
    stc=pickle.load(f)
    
fwd = mne.read_forward_solution(imp_dir.joinpath('fwd'))
    
src = fwd['src']


stc.save_as_volume('EEN_unmorphed.nii', src)  
