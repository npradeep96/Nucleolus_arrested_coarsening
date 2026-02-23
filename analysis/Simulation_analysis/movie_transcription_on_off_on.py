import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import h5py
import fipy as fp
import pandas as pd
import moviepy.editor as mp
import sys
sys.path.append('../')
import analysis.image_analysis as ia
import utils.simulation_helper as simulation_helper
import utils.file_operations as file_operations

from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
rc('text', usetex=False)
# rc('text.latex', preamble='\usepackage{color}')
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['font.size'] = 20
plt.rcParams["text.usetex"] = False


def write_movies_two_component_2d(path, stats_file, hdf5_file, movie_parameters, mesh, 
                                  t_off, t_on, fps=5):
    """Function that writes out movies of concentration profiles for 2 component simulations in 2D

    Args:
        path (string): Directory that contains the hdf5_file and input_parameters_file
        hdf5_file (string): Name of the hdf5 file that contains concentration profiles of the 2 components in 2D
        mesh (fipy.mesh): A fipy mesh object that contains mesh.x and mesh.y coordinates
        movie_parameters (dict): A dictionary that contains information on how to make the plots.
        fps (int): Frame per second to stitch together to make the movie. Default value is 3.
        t_off (float): Time at which transcription is turned off
        t_on (float): Time at which transcription is turned on again
    """

    # make directory to store the movies
    movies_directory = os.path.join(path, 'movies')
    try:
        os.mkdir(movies_directory)
        print("Successfully made the directory " + movies_directory + " ...")
    except OSError:
        print(movies_directory + " directory already exists")

    stats = pd.read_csv(stats_file, delimiter='\t')

    with h5py.File(os.path.join(path, hdf5_file), mode="r") as concentration_dynamics:
        # Read concentration profile data from files
        concentration_profile = []
        for i in range(int(movie_parameters['num_components'])):
            concentration_profile.append(concentration_dynamics['c_{index}'.format(index=i)])

        # Count the number of time points recorded for the movie
        t_max = concentration_profile[0].shape[0]
        for t in range(concentration_profile[0].shape[0]):
            # Check if we have reached the end of the movie
            flag = False
            zero_component_counter = 0
            for i in range(int(movie_parameters['num_components'])):
                if np.all(concentration_profile[i][t] == 0):
                    zero_component_counter = zero_component_counter + 1
            if zero_component_counter == int(movie_parameters['num_components']):
                t_max = t
                flag = True
            if flag:
                break

        for t in range(t_max):
            # Plot and save plots at each time point before stitching them together into a movie

            # Get upper and lower limits of the concentration values from the concentration profile data
            plotting_range = []
            for i in range(int(movie_parameters['num_components'])):
                # check if plotting range is explicitly specified in movie_parameters
                if 'c{index}_range'.format(index=i) in movie_parameters.keys():
                    plotting_range.append(movie_parameters['c{index}_range'.format(index=i)])
                else:
                    if i == 0:
                        min_value = np.min(concentration_profile[i][0])
                        max_value = np.max(concentration_profile[i][0])
                        plotting_range.append([min_value, max_value])
                    else:
                        min_value = np.min(concentration_profile[i][t])
                        max_value = np.max(concentration_profile[i][t])
                        plotting_range.append([min_value, max_value])

            # Generate and save plots
            fig, ax = plt.subplots(1, int(movie_parameters['num_components']), figsize=movie_parameters['figure_size'])
            for i in range(int(movie_parameters['num_components'])):
                # avg_value = np.average(concentration_profile[i][t])
                # normalized_concentration_profile = np.divide(concentration_profile[i][t], avg_value)
                cs = ax[i].tricontourf(mesh.x, mesh.y, concentration_profile[i][t],
                                       levels=np.linspace(plotting_range[i][0], plotting_range[i][1],
                                                          256),
                                       cmap=movie_parameters['color_map'][i])
                # ax[i].tick_params(axis='both', which='major', labelsize=20)
                ax[i].xaxis.set_tick_params(labelbottom=False)
                ax[i].yaxis.set_tick_params(labelleft=False)
                cbar = fig.colorbar(cs, ax=ax[i], ticks=np.linspace(np.ceil(plotting_range[i][0] * 1000) * 0.001,
                                                                    np.floor(plotting_range[i][1] * 1000) * 0.001,
                                                                    2))
                cbar.ax.tick_params(labelsize=30)
                ax[i].set_title(movie_parameters['titles'][i], fontsize=30)

                if stats['t'][t] < t_off or stats['t'][t] > t_on:
                    text = fig.text(0.20, 0.02, 'Transcription ON', horizontalalignment='center', wrap=True,
                                    fontsize=40)
                else:
                    text = fig.text(0.20, 0.02, 'Transcription OFF', horizontalalignment='center', wrap=True,
                                    fontsize=40)

                text = fig.text(0.83, 0.03, 'Time', horizontalalignment='center', wrap=True, fontsize=20)
                width = stats['t'][t] / stats['t'][t_max - 1]
                fig.patches.extend([plt.Rectangle((0.5, 0.02), 0.3, 0.05,
                                                  fill=False, color='k',
                                                  transform=fig.transFigure, figure=fig),
                                    plt.Rectangle((0.5, 0.02), 0.3 * width, 0.05,
                                                  fill=True, color='k',
                                                  transform=fig.transFigure, figure=fig)])

                fig.tight_layout(rect=(0, .1, 1, 1))

            fig.savefig(fname=movies_directory + '/Movie_step_{step}.png'.format(step=t), dpi=300, format='png')
            plt.close(fig)

    # Stitch together images to make a movie:

    def key_funct(x):
        return int(x.split('_')[-1].rstrip('.png'))

    file_names = sorted(list((file_name for file_name in os.listdir(movies_directory) if file_name.endswith('.png'))),
                        key=key_funct)
    file_paths = [os.path.join(movies_directory, f) for f in file_names]
    clip = mp.ImageSequenceClip(file_paths, fps=fps)
    clip.write_videofile(os.path.join(path, 'movies', 'Movie.mp4'), fps=fps)
    clip.close()

    # delete individual images
    for f in file_paths:
        os.remove(f)

if __name__ == "__main__":
    
    path = '/nfs/arupclab001/npradeep96/Nucleolus_arrested_coarsening/Model_Reformulation/FH_final_on_off_on_finer/M_1_24.0_M_2_2.4_beta_40.5_chi_40.0_N2_0.016_w2_0.004_K_0.05_c_initial_0.09'
    # You might want to change the above path variable to reflect the path where your simulation results are stored
    
    input_parameters_file = 'input_params.txt'
    stats_file = 'stats.txt'
    hdf5_file = 'spatial_variables.hdf5'
    
    movie_parameters = {'num_components': 2,
                        'color_map': ["Greens", "PuRd"],
                        'titles': ["FC Protein", "Nascent rRNA"],
                        'figure_size': [12, 6],
                        'c0_range': [0.0, 0.22],
                        'c1_range': [0.0, 0.011]}

    input_parameters_file = os.path.join(path, input_parameters_file)
    input_params = file_operations.input_parse(input_parameters_file)
    stats_file = os.path.join(path, stats_file)
    # Make a mesh of that geometry
    sim_geometry = simulation_helper.set_mesh_geometry(input_params)
    write_movies_two_component_2d(path=path, stats_file=stats_file, hdf5_file=hdf5_file,
                                movie_parameters=movie_parameters, mesh=sim_geometry.mesh)