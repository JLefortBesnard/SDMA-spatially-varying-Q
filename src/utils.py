from nilearn import plotting
import matplotlib.pyplot as plt
import numpy
import nibabel
import seaborn
import pandas
import os
import scipy


def plot_generated_data(generated_data, results_dir, fig_name):
    # #######################################
    # print("Plotting generated data")
    # #######################################
    print("Plotting data")
    plt.close('all')
    # f, axs = plt.subplots(1, len(generated_data.keys()), figsize=(len(generated_data.keys())*6, 6)) 
    f, axs = plt.subplots(1, len(generated_data.keys()), figsize=(len(generated_data.keys())*6, 6)) 
    for index, title in enumerate(generated_data.keys()):
        contrast_estimates = generated_data[title]
        mean = numpy.round(numpy.mean(contrast_estimates), 2)
        var = numpy.round(numpy.var(contrast_estimates), 2)
        spat_mat = numpy.corrcoef(contrast_estimates.T)
        corr_mat = numpy.corrcoef(contrast_estimates)
        seaborn.heatmap(contrast_estimates[:,:50], center=0, vmin=contrast_estimates.min(), vmax=contrast_estimates.max(), cmap='coolwarm', ax=axs[index],cbar_kws={'shrink': 0.5})
        # axs[index].title.set_text("{} data pipeline\nGenerated values (mean={}, var={})\nSpatial correlation={}\nPipelines correlation={}".format(title, mean, var, numpy.round(spat_mat.mean(), 2), numpy.round(corr_mat.mean(), 2)))
        # axs[index].title.set_text("{} data pipeline\nCorr = {}".format(title, numpy.round(corr_mat.mean(), 2)))
        scenario =["Independent", "Correlated", "Mixed"]
        axs[index].title.set_text("{} scenario\n(Corr = {})".format(scenario[index], numpy.round(corr_mat.mean(), 2)))
        axs[index].title.set_fontsize(20)
        axs[index].set_xlabel("J voxels", fontsize = 25)
        axs[index].set_ylabel("K pipelines", fontsize = 25)
        axs[index].tick_params(axis='y', labelrotation=0, labelsize=13)
        axs[index].tick_params(axis='x', labelrotation=0, labelsize=13)
        # Set colorbar tick label font size
        cbar = axs[index].collections[0].colorbar
        cbar.ax.tick_params(labelsize=13)  # Adjust labelsize here
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "{}.pdf".format(fig_name)))
    plt.close('all')
    print("Done plotting")

def distribution_inversed(J):
    distribution_inversed = []
    for i in range(J):
        distribution_inversed.append(i/(J+1))
    return distribution_inversed     

def minusLog10me(values):
    # prevent log10(0)
    return numpy.array([-numpy.log10(i) if i != 0 else 5 for i in values])


def plot_PP(Poster_results, figure_dir, corr, J, K):
     scenario =["Independent", "Correlated", "Mixed"]
     contrast_estimates = Poster_results[0][1]
     MA_outputs = Poster_results[0][0]
     p_cum = distribution_inversed(J)
     x_lim_pplot = -numpy.log10(1/J)
     MA_estimators = list(MA_outputs.keys())
     MA_estimators = list(MA_outputs.keys())[:]

     f, axs = plt.subplots(3, len(MA_estimators), figsize=(len(MA_estimators)*2.5, 8), sharey=True,sharex=True) 
     for row in range(3):
          for col, title in enumerate(MA_estimators):
               if title == "":
                    continue
               contrast_estimates = Poster_results[row][1]
               MA_outputs = Poster_results[row][0]
               simulation = Poster_results[row][2]
               # store required variables
               #  T_map, p_values, ratio_significance, verdict, _ = MA_outputs[title].values() # dangerous because dictionnary are not ordered
               T_map = MA_outputs[title]["T_map"]
               p_values = MA_outputs[title]["p_values"]
               ratio_significance = MA_outputs[title]["ratio_significance"]
               verdict = MA_outputs[title]["verdict"]

               # reformat p and t to sort and plot
               df_obs = pandas.DataFrame(data=numpy.array([p_values, T_map]).T, columns=["p_values", "T_values"])
               df_obs = df_obs.sort_values(by=['p_values'])
               # explected t and p distribution
               t_expected = scipy.stats.norm.rvs(size=J, random_state=0)
               # p_expected = 1-scipy.stats.norm.cdf(t_expected)
               p_expected = scipy.stats.norm.sf(t_expected)
               df_exp = pandas.DataFrame(data=numpy.array([p_expected, t_expected]).T, columns=["p_expected", "t_expected"])
               df_exp = df_exp.sort_values(by=['p_expected'])
               # Assign values back
               p_expected = df_exp['p_expected'].values
               t_expected = df_exp['t_expected'].values

               p_obs_p_cum = minusLog10me(df_obs['p_values'].values) - minusLog10me(p_cum)

               if row == 0:
                    axs[row][col].title.set_text(title)
                    axs[row][col].title.set_fontsize(15)
               elif row == 2:
                    # make pplot
                    axs[row][col].set_xlabel("-log10 cum p", fontsize=15)
               axs[row][col].plot(minusLog10me(p_cum), p_obs_p_cum, color='y')
               if col == 0:
                    axs[row][col].set_ylabel("{}\nscenario\n\nobs p - expt p".format(scenario[row]), fontsize=15)
               else:
                    axs[row][col].set_ylabel("")
               axs[row][col].axvline(-numpy.log10(0.05), ymin=-1, color='black', linewidth=0.5, linestyle='--')
               axs[row][col].axhline(0, color='black', linewidth=0.5, linestyle='--')

               ci = numpy.array([2*numpy.sqrt(p_c*(1-p_c)/J) for p_c in p_cum])
               p_obs_p_cum_ci_above = minusLog10me(numpy.array(p_cum)+ci) - minusLog10me(p_cum)
               p_obs_p_cum_ci_below = p_obs_p_cum_ci_above*-1
               axs[row][col].fill_between(minusLog10me(p_cum), p_obs_p_cum_ci_below, p_obs_p_cum_ci_above, color='b', alpha=.1)
               axs[row][col].set_xlim(0, x_lim_pplot)
               axs[row][col].set_ylim(-1, 1)
               color= 'green' if verdict == True else 'black'
               if color == 'black':
                    if ratio_significance > 5:
                         color= 'red'
               axs[row][col].text(1.5, -0.7, '{}%'.format(numpy.round(ratio_significance, 2)), color=color, fontsize=15)

               if row==0:
                    plt.tick_params(
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False)     # ticks along the bottom edge are off) 

     # plt.suptitle('P-P plots')
     
     if "\n" in simulation:
          simulation = simulation.replace('\n', '')
     simulation = simulation.replace(' ', '_')

     # plt.savefig("{}/pp_plot_OHBM_ABSTRACT.png".format(results_dir))
     plt.suptitle("{} voxels, {} pipelines, correlation between pipelines: {}".format(J, K, corr), fontsize=20)
     plt.tight_layout()
     plt.savefig(os.path.join(figure_dir, "J{}_K{}_Corr{}.pdf".format(J, K, corr)))
     plt.close('all')

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