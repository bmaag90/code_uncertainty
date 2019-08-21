# Code for "Enhancing Multi-Hop Sensor Calibration with Uncertainty Estimates"
This repository provides a simple framework to showcase the model presented in "Enhancing Multi-Hop Sensor Calibration with Uncertainty Estimates" (see [https://doi.org/10.3929/ethz-b-000352546](https://doi.org/10.3929/ethz-b-000352546)) that performs typical regression task while also providing epistemic and aleatoric uncertainty metrics.
The model is tested on artificially generated data, i.e. we sample a non-linear function with samples that are affected by input-dependent noise.

## How does the model work?

The model conists of two parts: 
1. a first ensemble of neural networks, which performs the regression task, i.e. estimate the underlying function `f(x)` using the noisy samples.
1. a second ensemble of neural networks, which learns the relationship (i.e. non-linear function) between the input `x` and the regression error of the first ensemble `(estimation - groundtruth)^2`

This allows us to calculate:
1. The final estimation of the underlying function: mean over all ensemble member outputs of the first ensemble
1. The epistemic uncertainty of the regression: standard deviation over all ensemble member outputs of the first ensemble
1. The aleatoric uncertainty of the regression: mean over all ensemble member outputs of the second ensemble
1. The epistemic uncertainty of the aleatoric uncertainty estimation: standard deviation over all ensemble member outputs of the second ensemble 

In order to save memory and minimize training efforts we use the network structure described in *Osband et al. Deep exploration via bootstrapped DQN. NIPS 2016*, i.e. we use shared hidden layers and multiple outputs that are trained individually by the bootstrapped datasets to create an ensemble.

More information about our model can be found in our [paper](https://doi.org/10.3929/ethz-b-000352546). Further information about:
1. Bootstrapping: *Efron and Tibshirani. An introduction to the bootstrap. 1994* 
1. Aleatoric uncertainty estimation: *Nix and Weigend. Learning local error bars for nonlinear regression. NIPS95*

## Code Structure
Following files are provided:
1. `RunExperiment.py`: Main script to perform an experiment, takes care of generating data, training the model and plotting the results. Run `python MultihopCalibration.py --config_file=config.json` to start a new experiment
1. `UncertaintyModel.py`: Tensorflow model
1. `DataCreator.py`: Creates the random data which is used to train the model
1. `ModelTrainer.py`: Framework to train the tensorflow model using different strategies such as learning-rate-decay or batch-size-increasing
1. `config.json:` Different configuration parameters used to perform the experiment
1. `example.png`: Plot of an example experiment (run default settings to re-create this result)

## Requirements
Implemented with:

1. python 2.7.12
1. numpy 1.14.5
1. tensorflow 1.10.1
1. matplotlib 2.1.1
