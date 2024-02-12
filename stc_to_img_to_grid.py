#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:00:38 2024

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

n_jobs = 10 #CHANGE
topdir = pathlib.PurePath('X:\\') # topdir = '/data/ML_MEG'
resultsdir = pathlib.PurePath(op.join(topdir, 'results_s_final'))
subj_dir = topdir.joinpath('NIH_hvmeg', 'derivatives', 'freesurfer', 'subjects')
vol_ratio_dir = topdir.joinpath('beamforming', 'epoched_3', 'vol', 'stc_ratio')
vol_image_dir = topdir.joinpath('beamforming', 'epoched_3', 'vol', 'image')

vol_images_dir = topdir.joinpath('beamforming', 'epoched_3', 'vol', 'images_select')
surf_images_dir = topdir.joinpath('beamforming', 'epoched_3', 'surf', 'images_select')

vol_nii_dir = topdir.joinpath('beamforming', 'epoched_3', 'vol', 'nii_ratio')
surf_ratio_dir = topdir.joinpath('beamforming', 'epoched_3', 'surf', 'stc_ratio')
surf_image_dir = topdir.joinpath('beamforming', 'epoched_3', 'surf', 'image')
subj_dir = pathlib.PurePath(op.join(topdir, 'NIH_hvmeg/derivatives/freesurfer/subjects'))
fname_fs_src = subj_dir.joinpath('fsaverage','bem', 'fsaverage-vol-5-src.fif')
src_fs = mne.read_source_spaces(fname_fs_src)

# =============================================================================
# Make an image given stc for surf
# =============================================================================

os.chdir(surf_ratio_dir)

for file in os.listdir(surf_ratio_dir):
    subject = file[:-10]
    with open(file, "rb") as f:
        stc_image=pickle.load(f)
    '''stc_image.data[stc_image.data > 0] = 0
    stc_image *= -1'''
    '''
    '''
    #Plotting to make the image
    os.chdir(surf_image_dir)
    
    kwargs = dict(
        initial_time=0,
        verbose=True,
        hemi = 'split',
        views=['lat', 'med'],
        size=(1600, 1600), 
    )
    brain = stc_image.copy().crop(tmin=0,tmax=0).plot(clim=dict(kind="value", pos_lims=[0.02, 0.04, 0.08]), **kwargs)    
    
    '''brain = stc_image.copy().crop(tmin=0,tmax=0).plot(hemi='split', views=['lat', 'med'], size=(1600, 1600), pos_lims=[0.02, 0.04, 0.08])'''
    
    
    brain.save_image(f'{subject}_image.png')
    brain.close()
    os.chdir(surf_ratio_dir)
    
# =============================================================================
# Make a surface image given stc for vol
# =============================================================================

os.chdir(vol_ratio_dir)

#Convert stc to Nifti
for file in os.listdir(vol_ratio_dir):
    subject = file[:-10]
    with open(file, "rb") as f:
        stc_image=pickle.load(f)
    os.chdir(vol_nii_dir)
    stc_image.save_as_volume(f'{subject}.nii', src_fs)  
    os.chdir(vol_ratio_dir)
    
#Make the image
os.chdir(vol_nii_dir)
for file in os.listdir(vol_nii_dir):
    print(file)
    subject = file[:-4]
    fsaverage = datasets.fetch_surf_fsaverage()
    curv_right = surface.load_surf_data(fsaverage.curv_right)
    curv_right_sign = np.sign(curv_right)
    
    curv_left = surface.load_surf_data(fsaverage.curv_left)
    curv_left_sign = np.sign(curv_left)
    
    texture_rh = surface.vol_to_surf(file, fsaverage.pial_right, inner_mesh = fsaverage.white_right)
    texture_lh = surface.vol_to_surf(file, fsaverage.pial_left, inner_mesh = fsaverage.white_left)
    threshold = 0.02
    
    fig, ax = subplots(2,2, subplot_kw={'projection': '3d'})
    
    ## Lateral
    plotting.plot_surf_stat_map(
        fsaverage.infl_left, texture_lh, hemi='left',
        title='Lateral left hemisphere', colorbar=True,
        threshold=threshold, bg_map=curv_right_sign, axes=ax[0,0]
    )
    
    plotting.plot_surf_stat_map(
        fsaverage.infl_right, texture_rh, hemi='right',
        title='Lateral right hemisphere', colorbar=True,
        threshold=threshold, bg_map=curv_right_sign, axes=ax[0,1]
    )
    
    ## Medial
    plotting.plot_surf_stat_map(
        fsaverage.infl_left, texture_lh, hemi='left',
        title='Medial left hemisphere', view='medial',colorbar=True,
        threshold=threshold, bg_map=curv_right_sign, axes=ax[1,0]
    )
    
    plotting.plot_surf_stat_map(
        fsaverage.infl_right, texture_rh, hemi='right',
        title='Medial right hemisphere', view='medial', colorbar=True,
        threshold=threshold, bg_map=curv_right_sign, axes=ax[1,1]
    )
    
    os.chdir(vol_image_dir)
    fig.savefig(f'{subject}_image.png')
    fig.clear()
    os.chdir(vol_nii_dir)
    
# =============================================================================
# Combines multiple images onto one grid for viewing ease
# =============================================================================
from PIL import Image, ImageDraw, ImageFont
import os

def resize_image(image, target_size):
    # Resize the image while preserving aspect ratio
    width, height = image.size
    aspect_ratio = width / height
    new_width = target_size[0]
    new_height = int(new_width / aspect_ratio)
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    return resized_image

def merge_images_with_text(vol_images_dir, surf_images_dir, topdir):
    images_a = sorted(os.listdir(vol_images_dir))
    images_b = sorted(os.listdir(surf_images_dir))
    num_images = len(images_a)
    assert num_images == len(images_b), "Number of images in folders must be the same."

    # Set a common target size for resizing
    target_size = (500, 500)  # Adjust as needed
    text_color = (0, 0, 0)  # Black text color
    font_size = 18
    line_spacing = 20  # Space between rows

    # Create a new image to hold the merged columns
    image_width = 2 * target_size[0]
    image_height = num_images * target_size[1] + (num_images - 1) * line_spacing
    merged_image = Image.new("RGB", (image_width, image_height), color="white")
    draw = ImageDraw.Draw(merged_image)

    y_offset = 0
    for img_a, img_b in zip(images_a, images_b):
        image_a = Image.open(os.path.join(vol_images_dir, img_a))
        image_b = Image.open(os.path.join(surf_images_dir, img_b))
        resized_a = resize_image(image_a, target_size)
        resized_b = resize_image(image_b, target_size)

        # Paste resized images with blank space in between
        merged_image.paste(resized_a, (0, y_offset))
        y_offset += target_size[1] + line_spacing
        merged_image.paste(resized_b, (target_size[0], y_offset))
        y_offset += target_size[1]

    # Add text to the blank space
    font = ImageFont.truetype("arial.ttf", font_size)  # Use any font you like
    text_position = (10, image_height - line_spacing)  # Adjust text position
    draw.text(text_position, "xxxx", fill=text_color, font=font)

    # Save the merged image
    merged_image.save(os.path.join(topdir, "merged_image_with_text.jpg"))








def resize_image(image, target_size):
    return image.resize(target_size, Image.ANTIALIAS)

def merge_images(folder_a, folder_b, output_folder):
    # Get the list of image filenames in both folders
    images_a = sorted(os.listdir(vol_images_dir))
    images_b = sorted(os.listdir(surf_images_dir))

    # Ensure that both folders have the same number of images
    num_images = len(images_a)
    assert num_images == len(images_b), "Number of images in folders must be the same."

    # Set a common target size for resizing
    target_size = (500, 500)  # Adjust as needed

    # Create a new image to hold the merged columns
    image_width = 2 * target_size[0]
    image_height = num_images * target_size[1]
    merged_image = Image.new("RGB", (image_width, image_height))

    # Paste resized images from folder_a into the left column
    y_offset = 0
    for img_a, img_b in zip(images_a, images_b):
        image_a = Image.open(os.path.join(vol_images_dir, img_a))
        image_b = Image.open(os.path.join(surf_images_dir, img_b))
        resized_a = resize_image(image_a, target_size)
        resized_b = resize_image(image_b, target_size)
        merged_image.paste(resized_a, (0, y_offset))
        merged_image.paste(resized_b, (target_size[0], y_offset))
        y_offset += target_size[1]

    output_folder = topdir

    # Save the merged image
    merged_image.save(os.path.join(output_folder, "merged_image.jpg"))
































os.chdir(surf_images_dir)

# Define the number of images per row and the spacing between them
columns = 1
space = 20

# Load the images from a list of file names
images2 = os.listdir(surf_images_dir)

images = [Image.open(x) for x in images2]

# Get the maximum width and height of the images
widths, heights = zip(*(i.size for i in images))
max_width = max(widths)
max_height = max(heights)

# Create a new image with a white background
total_width = max_width * columns + space * (columns - 1)
total_height = max_height * ((len(images)) // columns + 1) + space * ((len(images)) // columns)
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
os.chdir("X:\\beamforming\\epoched_3\\final_res")
new_im.save("surf_images.jpg")