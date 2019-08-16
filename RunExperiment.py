from DataCreator import DataCreator
from ModelTrainer import ModelTrainer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
import json

'''
Main script to run an experiment
'''

'''
Get the config (json) file, see "config.json" for default one
'''
parser = argparse.ArgumentParser(description='Parse location of config file (json).')
parser.add_argument('--config_file', type=str, default='config.json',
                    help='path to json config file, see config.json for default')

args = parser.parse_args()
with open(args.config_file) as json_data_file:
    config = json.load(json_data_file)

'''
Create random data
'''
myDataCreator = DataCreator(config["number_of_samples"],x_min=-2*np.pi, x_max=2*np.pi,random_seed=config["random_seed"])
myDataCreator.create_datapoints()
x, y, f, noise = myDataCreator.get_data()

'''
Define the interval where the training data is taken from to train the model (default: -pi:pi)
'''
x_train_min = -1*np.pi
x_train_max = np.pi
idx_train = np.where(np.logical_and(x>=x_train_min, x<=x_train_max))

'''
Create the model trainer
'''
myModelTrainer = ModelTrainer(config["nn_sizes"],
    config["learning_rate"],
    config["ensemble_size"],
    config["beta_pred"],
    config["beta_unc"],
    config["random_seed"])

'''
Train the model
'''
myModelTrainer.train_model(x[idx_train[0]],y[idx_train[0]],config["epochs"],config["batch_size"],config["learning_decay"],config["max_stop_cnt"],config["update_step"])

'''
Evaluate the model on all the data for plotting
'''
PredMat, UncMat = myModelTrainer.eval_model(x)

'''
Calculate the final prediction (mean over all ensemble members of the prediciton path), 
the epistemic uncertainty (standard deviation over all ensemble members of the prediciton path) and
the aleatoric (mean over all ensemble members of the uncertainty path) 
'''
Prediction = np.mean(PredMat,axis=1)
EpistemicUncertainty = np.std(PredMat,axis=1)
AleatoricUncertainty = np.mean(UncMat,axis=1)


'''
Plot the results
'''
plt.figure(figsize=(16,8))
plt.subplot(2, 1, 1)
plt.scatter(x,y,c=[0.5,0.5,0.5],label='Samples')
plt.plot(x,f,'k',label='Underlying Function')
plt.plot(x,Prediction,'b--',label='Prediction')
axes = plt.gca()
ylim = axes.get_ylim()
axes.add_patch(Rectangle((x_train_min,ylim[0]),abs(x_train_max-x_train_min),abs(ylim[1]-ylim[0]),alpha=0.2, facecolor='y',label='Training Area'))
axes.legend()
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(2, 1, 2)
plt.plot(x,EpistemicUncertainty,'g-.',label='Epistemic Uncertainty')
plt.plot(x,AleatoricUncertainty,'r-.',label='Aleatoric Uncertainty')
axes = plt.gca()
ylim = axes.get_ylim()
axes.add_patch(Rectangle((x_train_min,ylim[0]),abs(x_train_max-x_train_min),abs(ylim[1]-ylim[0]),alpha=0.2, facecolor='y',label='Training Area'))
axes.legend(['Aleatoric Uncertainty','Epistemic Uncertainty','Training Area'])
plt.xlabel('x')
plt.ylabel('Uncertainty')
plt.show()


