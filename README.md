# Code for "Enhancing Multi-Hop Sensor Calibration with Uncertainty Estimates"
This repository provides a simple framework to showcase the model presented in "Enhancing Multi-Hop Sensor Calibration with Uncertainty Estimates" that performs typical regression task while also providing epistemic and aleatoric uncertainty metrics.
The model is tested on artificially generated data, i.e. we sample a non-linear function with samples that are affected by input-dependent noise.

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
