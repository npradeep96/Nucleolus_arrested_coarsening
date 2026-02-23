import scipy.stats
import numpy as np

from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from .distributions import *

# Set the x labels for the plot
def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)

def plot_pdf_normalized(ax, data, distribution, xlabel, bins = 40):
    x = np.linspace(data.min(), data.max(), 100)
    ax.hist(data, bins = bins, density=True, alpha=0.6, color='g');
    ax.plot(x, distribution.pdf(x), color="black", lw=4)
    
    test = scipy.stats.cramervonmises(data, cdf=distribution.cdf)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.text((xmin+xmax)/2,ymin - 0.2*(ymax-ymin), "statistic: " + str(np.format_float_scientific(test.statistic, precision=2)), horizontalalignment='center')
    ax.text((xmin+xmax)/2,ymin - 0.25*(ymax-ymin), "p-value: " + str(np.format_float_scientific(test.pvalue, precision=2)), horizontalalignment='center')
    ax.set_xlabel(xlabel)
    ax.set_ylabel("probability density")

"""
Function to fit the three different distributions given the data to distributions with parameters
that describe them including the mean and the variance, also taking into account the minimum size thresholds.
This plots the fit cumulative distributions to assess the fit, and also returns the fitting statistics
"""
def plot_cdf_cutoff(ax, data):
    x = np.linspace(data.min(), data.max(), 100)
    ax.hist(data, bins = 40, cumulative=True, density=True, alpha=0.6, color='g');
    
    # Fit the three distributions to data
    best_lsw, best_lsw_cdf, best_lsw_pvalue = fit_lsw(data)
    best_normal, best_normal_cdf, best_normal_pvalue = fit_normal(data)	
    best_log_normal, best_log_normal_cdf, best_log_normal_pvalue = fit_log_normal(data)
       
    ax.plot(x, best_lsw_cdf(x), label='LSW', c='k', linestyle='--')
    ax.plot(x, best_normal_cdf(x), label='Normal', c='k')
    ax.plot(x, best_log_normal_cdf(x), label='Log Normal', c='k', linestyle=':')
    
    ax.set_xlabel(r'R ($\mu$m)')
    ax.set_ylabel(r'cumulative distribution function')
    ax.legend(loc='lower right', frameon=False)

def plot_violin(ax, data, ylabel, ylims):

    # Plot violinplot
    parts = ax.violinplot(data, vert=True, widths=0.8, showmeans=True, showextrema=False)
    
    parts['cmeans'].set_color(['k']*len(data))
    for pc in parts['bodies']:
        pc.set_alpha(1)
        pc.set_facecolor('none')
        pc.set_edgecolor('black')
        pc.set_linewidth(2.0)
        
    set_axis_style(ax, list(data.index))

    ax.set_ylabel(ylabel)
    ax.set_ylim(ylims)
    
    # Plot the scatter plot of data
    for i, (key, val) in enumerate(data.items()):
        # Add some random "jitter" to the x-axis
        x = np.random.normal(i+1, 0.1, size=len(val))
        ax.scatter(x, val, color='steelblue', alpha=1.0, s=1.0, zorder=-100)

        ymin, ymax = ax.get_ylim()
        ax.text(i+1, ymin - 0.1 * (ymax-ymin), str(np.round(np.mean(val),2)) + "±" + str(np.round(np.std(val),2)), horizontalalignment='center',)


def plot_density(fig, ax, data, R, L):

    # Create data: 200 points
    x, y = np.stack(data).T
    mask = (x > -L) & (x < L) & (y > -L) & (y < L)
    
    # Create some dummy data
    rvs = np.stack([np.concatenate([x[mask],x[mask]]), np.concatenate([y[mask],-y[mask]])]).T
    kde = gaussian_kde(rvs.T, bw_method="silverman")
    
    # Regular grid to evaluate kde upon
    
    nbins = 100
    xi, yi = np.mgrid[x[mask].min():x[mask].max():nbins*1j, y[mask].min():y[mask].max():nbins*1j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
    
    cplot = ax.contourf(xi, yi, zi.reshape(xi.shape), zorder = -100, cmap='Greys')
    #ax.contour(xi, yi, zi.reshape(xi.shape), colors="black", zorder = -50)
    ax.plot([3*np.cos(np.pi/3), 0,3*np.cos(-np.pi/3)], [3*np.sin(np.pi/3), 0, 3*np.sin(-np.pi/3)], color="black", lw=1)
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    ax.set_aspect(1)
    
    ax.set_xlabel(r' relative position ($\mu$m)')
    ax.set_ylabel(r' relative position ($\mu$m)')
    
    ax.scatter([0, R], [0, 0], color="black")
    ax.text(1.2*R, 0, "nearest\nneighbor", color="black", horizontalalignment="left", verticalalignment="center")
    ax.text(1.2*R, 1.0, "+60°", color="black", horizontalalignment="left", verticalalignment="center", rotation=60)
    ax.text(1.2*R, -1.0, "-60°", color="black", horizontalalignment="left", verticalalignment="center", rotation=-60)
    
    circle1 = plt.Circle((0, 0), R, color='black', fill=False, lw=1)
    ax.add_patch(circle1)
    
    cbar = fig.colorbar(cplot, ax=ax)
    cbar.set_label("probability density")
    
    ax.set_title("distribution of next-nearest neighbors")