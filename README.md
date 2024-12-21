# README

## Overview

This script, `prep_limit_new.py`, is designed to prepare data for fitting, bias, and limit calculations using Gaussian Processes (GP). It performs various tasks such as loading data, optimizing parameters, generating toy datasets, and running bias studies.

(The `prep_limit_mc.py` script works the same as here, but it takes in the histogram for fititng along with the MC component of backgorund, rest of the things are same. In future the two scripts will be merged into one.)

## Requirements

- Python 3.x
- ROOT
- NumPy
- SciPy
- Matplotlib
- mplhep
- iminuit
- argparse

Requirements can also be installed through conda with the `environment.yml` file as follows
```
conda env create -f environment.yml
```

## Usage

The script can be run with the following command:

```bash
python prep_limit_new.py [OPTIONS]
```

### Options
- `--input_file`: Path to the input ROOT file (default: resolved2016_reg2.root).
- `--hist_name`: Name of the histogram in the ROOT file (default: h_mass).
- `--length_scale`: Length scale parameter for the GP (default: 100).
- `--variance`: Variance parameter for the GP (default: 10).
- `--mean`: Mean value for the signal (default: 10).
- `--sigma`: Sigma value for the signal (default: 10).
- `--rate_uc`: Rate uncertainty (default: 0.11).
- `--mean_err`: Error in the mean value (default: 10).
- `--sigma_err`: Error in the sigma value (default: 10).
- `--nwalkers`: Number of walkers for MCMC (default: 12).
- `--steps`: Number of steps for MCMC (default: 100000).
- `--sig_strength`: Signal strength (default: 10).
- `--run_bias`: Whether to run bias studies (default: False).
- `--submit_condor`: Whether to submit jobs to Condor (default: False).
- `--rebin`: Rebin factor for the histogram (default: 1).
- `--ntoys`: Number of toy datasets to generate (default: 500).
- `--tag`: Tag for output files (default: 3j).

### Example

```bash
python prep_limit_new.py --input_file resolved2016_reg2.root --hist_name h_mass --length_scale 100 --variance 10 --mean 10 --sigma 10 --rate_uc 0.11 --mean_err 10 --sigma_err 10 --nwalkers 12 --steps 100000 --sig_strength 10 --run_bias --submit_condor --rebin 1 --ntoys 500 --tag 3j
```

## Output

The script generates the following output files:

- `fit_<tag>_<mean>.pdf`: Plot of the fit results.
- `fit_toy_<tag>_<mean>.pdf`: Plot of the toy dataset fit results.
- `toys_<mean>_<tag>_3j.npz`: Numpy archive containing toy datasets.
- `sig_strength.pdf`: Plot of the signal strength profile.
- `bias3_<mean>_<lm>_3j.pdf`: Plot of the bias study results.
- `ss3_<mean>_<lm>_3j.pdf`: Plot of the signal strength bias study results.

## Condor Submission

If the `--submit_condor` option is used, the script will generate and submit a Condor job script for running the limit calculations.

### Interactive run for limit calculation

To run the limit calculation interactively, use the following command:

```bash
python run_limits_new.py --toyn=1 --input_file=toys_<mean>_<tag>_3j.npz --length_scale=<length_scale> --variance=<variance> --length_scale_err=0 --variance_err=0 --mean=<mean> --sigma=<sigma> --rate_uc=<rate_uc> --mean_err=<mean_err> --sigma_err=<sigma_err> --nwalkers=<nwalkers> --steps=<steps> --sig_strength=<sig_strength> --show_result=True
```
