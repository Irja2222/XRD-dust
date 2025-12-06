import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import container




# Campioni.
#####################################################
samples = ["A01_Cs2AgInCl6", "A02_Cs2Na0.2Ag0.8InCl6", "A04_Cs2Na0.6Ag0.4InCl6", "A10_Cs2NaInCl6"]
samples_plot = ["$Cs_2AgInCl_6$", "$Cs_2Na_{{0.2}}Ag_{{0.8}}InCl_6$", "$Cs_2Na_{{0.6}}Ag_{{0.4}}InCl_6$", "$Cs_2NaInCl_6$"]


#####################################################
# Upload datasets.
# col = 0: 2*theta (deg)
# col = 1: experimental data (counts)
# col = 2: fitted data (counts)
#####################################################
folder = "XRD-dust/Datasets/"
files = os.listdir(folder)

# Cut datasets and fits.
#####################################################
dataset_names = []
datasets_fit = []
for file in files:
    if file.endswith(".txt"):
        dataset_names.append(file)
        datasets_fit.append(np.loadtxt(folder + file, skiprows = 15, usecols = (0, 1, 2)))

# Raw datasets.
#####################################################
dataset_compl_names = []
datasets_compl = []
for file in files:
    if file.endswith(".xy") and not "cut" in file:
        dataset_compl_names.append(file)
        datasets_compl.append(np.loadtxt(folder + file, skiprows = 1))


# Plot datasets.
#####################################################
def plot(dataset_raw, dataset_fit, sample, sample_plot):
    my_dpi = 192
    tnrfont = {'fontname':'Times New Roman'}
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.fancybox"] = False

    _, ax = plt.subplots(nrows = 2, ncols = 1, 
                            gridspec_kw = {'width_ratios': [1], 'height_ratios': [4, 1], 'wspace': 0, 'hspace': 0.1},
                            num = sample,
                            figsize = (4000/my_dpi, 3000/my_dpi),
                            dpi = my_dpi)
    #########
    # [0,0] #
    #########
    ax[0].plot(dataset_fit[:, 0], dataset_fit[:, 2],
            color = "red",
            label = "Rietveld fit",
            linewidth = 1.5,
            zorder = 2)
    ax[0].scatter(dataset_raw[:, 0], dataset_raw[:, 1],
            color = "black",
            label = "experimental data",
            marker = 'o',
            s = 60, 
            edgecolor = "black",
            facecolor = "white",
            linewidth = 2,
            zorder = 1)
    ax[0].text(0.67, 0.85,
             "Rietveld refinement of a\nBragg-Brentano XRD diffractogram\nfor a " + sample_plot + " powder sample",
             **tnrfont,
             fontsize = 50,
             horizontalalignment = 'center',
             verticalalignment = 'center',
             transform = ax[0].transAxes,
             bbox = dict(facecolor = 'white', alpha = 1),
             zorder = 1)
    label_fontsize = 48
    x_min, x_max = np.min(dataset_raw[:, 0]), np.max(dataset_raw[:, 0])
    ax[0].set_xlim(x_min, x_max)
    ax[0].set_xticks([])
    ax[0].set_ylabel("counts",
                fontsize = label_fontsize,
                **tnrfont)
    y_min, y_max = np.min(dataset_raw[:, 1]), np.max(dataset_raw[:, 1] + 15)
    ax[0].set_ylim(y_min, y_max)
    yticks = np.linspace(y_min, y_max, 6)
    ytick_names = []
    for ytick in yticks:
        ytick_names.append(int(ytick))
    ax[0].set_yticks(yticks, ytick_names)
    ax[0].tick_params(axis = 'y', labelsize = 34)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    handles, labels = ax[0].get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax[0].legend(handles, labels, loc = "center right", prop = {'size':35})
    #########
    # [1,0] #
    #########
    ax[1].plot(np.linspace(x_min, x_max, 2), np.zeros((2, )), '--',
                    color = "darkorange",
                    label = "expected residuals",
                    linewidth = 2,
                    zorder = 2)
    ax[1].plot(dataset_fit[:, 0], dataset_fit[:, 2] - dataset_fit[:, 1],
            color = "black",
            label = "computed residuals",
            linewidth = 1.5,
            zorder = 2)
    ax[1].set_xlabel("$2 \\theta \\ \\lbrack deg \\rbrack$",
                fontsize = label_fontsize,
                **tnrfont)
    ax[1].set_xlim(x_min, x_max)
    xticks = np.linspace(x_min, x_max, 11)
    ax[1].set_xticks(xticks)
    y_min, y_max = np.min(dataset_fit[:, 2] - dataset_fit[:, 1]), np.max(dataset_fit[:, 2] - dataset_fit[:, 1])
    ax[1].set_ylim(y_min, y_max)
    yticks = np.linspace(y_min, y_max, 4)
    ytick_names = []
    for ytick in yticks:
        ytick_names.append(round(ytick, 1))
    ax[1].set_yticks(yticks, ytick_names)
    ax[1].tick_params(axis = 'x', labelsize = 34, pad = 10)
    ax[1].tick_params(axis = 'y', labelsize = 34)
    handles, labels = ax[1].get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax[1].legend(handles, labels, loc = "best", prop = {'size':20})
    plt.savefig(folder + sample + ".pdf",
                dpi = my_dpi, bbox_inches = 'tight')



#####################################################
# Lattice parameters.
#####################################################

# Fit goodness [%].
#####################################################
Rwp_maud = np.array([9.747296, 9.15295, 10.356591, 13.003058])
w = []
R_wp = []
R_exp = []
chis_squared = []
for sample in range(len(samples)):
    w.append(1 / np.sqrt(datasets_fit[sample][:, 1]))
    R_wp.append(np.sqrt(np.sum((w[sample] * (datasets_fit[sample][:, 1] - datasets_fit[sample][:, 2]))**2) / np.sum((w[sample] * datasets_fit[sample][:, 1])**2)))
    R_exp.append(np.sqrt(len(datasets_fit[sample][:, 1]) / np.sum((w[sample] * datasets_fit[sample][:, 1])**2)))
    chis_squared.append((R_wp[sample]/R_exp[sample])**2)
    print("For " + samples[sample] + ", R_wp = {}, R_exp = {} and so chi^2 = {}.".format(round(R_wp[sample], 3), round(R_exp[sample], 3), round(chis_squared[sample], 3)))

# Concentrantions [%].
#####################################################
C_Ag_th = np.array([1, 1, 0.8, 0.4])
C_Ag = np.array([1, 1, 0.8214405, 0.35777777])
C_Ag_err = np.array([0, 0, 0.0029929676, 0.028463716])
C_Na_th = np.array([1, 1, 0.2, 0.6])
C_Na = np.array([1, 1, 0.204556, 0.6448477])
C_Na_err = np.array([0, 0, 0.0045996034, 0.11161944])

# Cell lengths [A°].
#####################################################
a = np.array([10.480548, 10.489761, 10.509253, 10.531843])
a_err = np.array([2.3117219e-4, 2.2038945e-4, 5.212044e-4, 2.971206e-4])

# Vegard's law [A°].
#####################################################
x = (a[1:3] - a[0]) / (a[-1] - a[0])
x_err = np.sqrt((a_err[1:3] / (a[-1] - a[0]))**2 + 
                (a_err[0] * (a[1:3] - a[-1]) / (a[-1] - a[0])**2)**2 +
                (a_err[-1] * (a[1:3] - a[0]) / (a[-1] - a[0])**2)**2)
x_discr = np.abs(x + x_err - C_Na_th[2:]) / C_Na_th[2:] * 100
for hybrid in range(len(x)):
    print("For " + samples[hybrid + 1] + ", x = ({} +- {}) %, with a discrepancy of {}%.".format(round(x[hybrid], 3), round(x_err[hybrid], 3), round(x_discr[hybrid], 2)))








if __name__ == "__main__":
    for sample in range(len(samples)):
        plot(datasets_compl[sample], datasets_fit[sample], samples[sample], samples_plot[sample])