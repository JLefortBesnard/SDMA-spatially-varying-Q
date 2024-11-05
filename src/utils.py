from nilearn import plotting
import matplotlib.pyplot as plt
import numpy
import nibabel
import seaborn
import pandas
import os
import scipy


### Function to plot SDMA p statistics into MNI brain 
def plot_significant_Pvalues_into_MNI(MA_outputs, hyp, MA_estimators_names, results_dir, masker):
    plt.close('all')
    f, axs = plt.subplots(len(MA_estimators_names[1:]), 1, figsize=(8, len(MA_estimators_names[1:])*8/5))
    for row, title in enumerate(MA_estimators_names[1:]):
        T_map=MA_outputs[title]['T_map']
        p_values=MA_outputs[title]['p_values']
        # compute ratio of significant p-values
        ratio_significance = MA_outputs[title]['ratio_significance']

        # back to 3D
        # p_brain_sign = masker.inverse_transform(p_stat)
        p_brain = masker.inverse_transform(p_values)
        t_brain = masker.inverse_transform(T_map)

        # apply threshold
        pdata = p_brain.get_fdata()
        tdata = t_brain.get_fdata()
        threshdata = (pdata <= 0.05)*tdata #0.05 is threshold significance
        threshimg = nibabel.Nifti1Image(threshdata, affine=t_brain.affine)
        long_title = title + ', {}%'.format(numpy.round(ratio_significance, 2))
        if "\n" in long_title:
            long_title = long_title.replace('\n', '')
        plotting.plot_stat_map(threshimg, annotate=False, threshold=0.1,vmax=8, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52), display_mode='z', cmap='Reds', axes=axs[row])
        axs[row].set_title(long_title)
    plt.suptitle('Hypothesis {}'.format(hyp))
    plt.savefig(os.path.join(results_dir, "thresholded_map_hyp{}.png".format(hyp)))
    plt.close('all')