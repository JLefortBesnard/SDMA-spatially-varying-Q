
import os
import numpy
import nibabel
from nilearn.input_data import NiftiMasker
from nilearn import image, plotting

import matplotlib.pyplot as plt
import seaborn as sns


##################
# Apply SDMA in all NARPS hypothesis outcomes
##################

# path to partiticipants mask
participants_mask_path = os.path.join("results", "NARPS", "masking", "participants_mask.nii")
participant_mask = nibabel.load(participants_mask_path)
# path to resampled NARPS data
data_path = os.path.join("data", "NARPS")
# folder to store results
results_dir = os.path.join("results", "NARPS")
figures_dir = os.path.join("figures", "NARPS")

# hypotheses = {1: '+gain: equal indiff',
#               2: '+gain: equal range',
#               3: '+gain: equal indiff',
#               4: '+gain: equal range',
#               5: '-loss: equal indiff',
#               6: '-loss: equal range',
#               7: '+loss: equal indiff',
#               8: '+loss: equal range',
#               9: '+loss:ER>EI'}

# save mask for inverse transform
masker = NiftiMasker(
    mask_img=participant_mask)


# compute and plot significant p values for each hypothesis and SDMA methods
hyp = 1
print('*****Running SDMA in NARPS hyp ', hyp, '*****')
results_dir_hyp = os.path.join(results_dir, "hyp{}".format(hyp), "smoothed_Q")
figures_dir_hyp = os.path.join(figures_dir, "hyp{}".format(hyp), "smoothed_Q")

# load data
data_path = os.path.join("data", "NARPS")
resampled_maps_per_team = numpy.load(os.path.join(data_path, "Hyp{}_resampled_maps.npy".format(hyp)), allow_pickle=True).item()
resampled_maps = masker.fit_transform(resampled_maps_per_team.values())
team_names = list(resampled_maps_per_team.keys())
print("Data loading + masking DONE")
img_demo_per_fwhm = {}
img_demo_per_fwhm['original_img_demo']=resampled_maps_per_team['4988_98BT']
cov_matrices = {}
cov_matrices['smooth_0_cov_matrix']=numpy.cov(resampled_maps, bias=False)
resampled_maps = None # empyting RAM memory

smoothing_values = [2, 4, 6, 8, 10]  # Example smoothing values in mm
# Dictionary to store smoothed images
# Apply smoothing for each smoothing value
for i, fwhm in enumerate(smoothing_values):
    resampled_maps_per_team = numpy.load(os.path.join(data_path, "Hyp{}_resampled_maps.npy".format(hyp)), allow_pickle=True).item()
    for team_name in resampled_maps_per_team.keys():
        print('smoothing ', team_name, ' at fwhm=', fwhm)
        smoothed_team_map = image.smooth_img(resampled_maps_per_team[team_name], fwhm=fwhm)
        resampled_maps_per_team[team_name] = smoothed_team_map
    resampled_maps = masker.fit_transform(resampled_maps_per_team.values())
    cov_matrices['smoothed_{}_cov_matrix'.format(fwhm)] =numpy.cov(resampled_maps, bias=False)
    img_demo_per_fwhm['smoothed_{}_img_demo'.format(fwhm)]=resampled_maps_per_team['4988_98BT']
    resampled_maps = None # empyting RAM memory


### PLOT SMOOTHED MAPS ####
###########################

# Set up the figure for horizontal subplots
fig, axes = plt.subplots(1, len(img_demo_per_fwhm), figsize=(15, 5))

# Iterate over each image in the dictionary
for ax, (name, img) in zip(axes, img_demo_per_fwhm.items()):
    # Get the image data
    img_data = img.get_fdata()
    
    # Take the middle slice in the z-dimension (you can adjust this as needed)
    middle_slice = img_data.shape[2] // 2
    slice_data = img_data[:, :, middle_slice]
    
    # Plot the slice
    ax.imshow(slice_data.T, cmap='gray', origin='lower')
    ax.set_title(name[:-9])
    ax.axis('off')

# Adjust layout for better spacing
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "hyp1", "smoothed_Q_maps.png"))
plt.close("all")


### PLOT COV MATRIX OF SMOOTHED MAPS ####
#########################################

ticks_label = [i for i in range(0, len(team_names))]

# Create a plot with 5 subplots in a row, using gridspec for more control over subplot sizes
fig = plt.figure(figsize=(40, 5))
gs = fig.add_gridspec(1, 6, wspace=0.1, width_ratios=[1, 1, 1, 1, 1, 1])  # Last subplot gets more space
# Iterate over each covariance matrix
for i, key in enumerate(cov_matrices.keys()):
    ax = fig.add_subplot(gs[0, i])
    sns.heatmap(cov_matrices[key], annot=False, cmap='RdBu_r', ax=ax, cbar=True, square=True,
                xticklabels=ticks_label, yticklabels=ticks_label if i == 0 else False,
                cbar_kws={'shrink': 0.3})  # Colorbar only on the last plot

    # Rotate x-axis labels on all subplots and reduce font size
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=4)
    if i == 0:
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=4)
    
    # Set title for each subplot
    ax.set_title(key[:-11])

# Adjust layout to avoid overlap and ensure equal subplot size
plt.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.01)
plt.savefig(os.path.join(figures_dir, "hyp1", "smoothed_Q_cov_mat.png"))
plt.close("all")
# # compute covariance matrix
# resampled_maps, resampled_maps_per_team = None, None # empyting RAM memory






