## Structure of the directory

This Github repo contains the code associated with simulations and data analysis for the pre-print **Active RNA synthesis patterns nuclear condensates** 
 (https://www.biorxiv.org/content/10.1101/2024.10.12.614958v1). 
 
 The repository is organized as follows:
1. ```scripts/``` contains the python code to numerically integrate the partial differential equations in the model and get the concentration profiles of the FC proteins and RNA as a function of space and time. The numerical integration is done by ```run_simulation.py``` and ```sweep_parameter.py``` is a wrapper script around ```run_simulation.py``` to enable parameter sweeps by running this script for different values of a parameter present in ```inputs/sweep_parameters.txt```.
2. The directory ```inputs``` contains two text files: ```input_parameters.txt``` which contains parameters related to the free energy, kinetics, initial conditions, geometry, and numerical parameters for integration that are inputs to ```scripts/run_simulation.py```. The file ```scripts/sweep_parameters``` can contain a parameter name and a list of parameter values that are used to run parameter sweeps over those values of the parameter.
3. The directory ```utils``` contains helper functions that are used by ```scripts/run_simulations.py```.
4. The directory ```analysis`` contains: (i) Data and jupyter notebooks for analysis of microscopy images in ```Microscopy_droplet_size_analysis```. (ii) Scripts to generate movies from simulation data in ```Simulation_analysis```

## Create conda environment and install libraries from the requirements.txt

Run the following command in your shell to set up the libraries that are necessary to run the simulation scripts ```scripts/run_simulation.py``` and ```scripts/sweep_parameters.py```:

```
conda create --name <Your_Environment_Name> python=2.7
conda activate <Your_Environment_Name>
pip install -r requirements.txt
```

## Run numerical simulations

The file ```inputs/input_parameters.txt``` contains parameters necessary to run the numerical simulation and is an input to ```scripts/run_simulation.py```. To run simulation, from the root directory of the repo, simply run the follwoing commands in your shell:

```
source activate <Your_Environment_Name>
python ./scripts/run_simulation.py --i ./inputs/input_parameters.txt --o <Output_directory_to_write_simulation_data>
```

You might want to set up a slurm (or pbs) script that runs the above command rather than run it on the shell as it takes some time. An example shell script for slurm is also provided in ```scripts/run_simulation/slurm```, that you can modify for your HPC cluster to submit jobs if it runs slurm. The file ```inputs/sweep_parameters.txt``` contains a parameter name and a list of values to run parameter sweeps and is an input to ```scripts/sweep_parameters.py```, which in turn submits a slurm job for each parameter value. To run parameter sweeps, from the root directory of the repo, simply run the follwoing commands in your shell:

```
python sweep_parameters.py --i ../inputs/input_parameters.txt --s ../inputs/sweep_parameters.txt --o <Output_directory_to_write_simulation_data>
```

The output files by running the above with the values of parameters present in ```inputs/input_parameters.txt``` and ```inputs/sweep_parameters.txt``` provided with this repo are present in the directory ```test_run``` to help you verify if the code is running correctly in your system.

## Make movies of simulations

To make movies after the simulation has completed, run the following command from the root directory of this repo:

```
python ./analysis/Simulation_analysis/make_movies.py --i <Output_path_with_simulation_data> --r <Regular_expression_for_directory_name> --h spatial_variables.hdf5 --p input_params.txt --m movie_parameters.txt
```

You can change ```analysis/Simulation_analysis/movie_parameters.txt``` to customize the looks of your movies. 

