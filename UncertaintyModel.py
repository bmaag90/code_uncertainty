import tensorflow as tf
import numpy as np


class UncertaintyModel():
    '''
    UncertaintyModel: 
    This model consists of two optimation paths: One used to perform the regression i.e. the PREDICTION task
    and one to calculate the first type of uncertainty, i.e. the aleatoric UNCERTAINTY
    Both paths are built as ensembles, where there are one or multiple shared hidden layers and multiple outputs which are trained individually. 
    The individual training can be done via bootstrapping, i.e. each output is trained with a bootstrapped dataset. 
    The hidden layers are shared to optimize training and memory requirements. [See also: Osband et al. Deep exploration via bootstrapped DQN, NIPS 2016]

    The model needs following parameters:
    nn_sizes: The size of the model, i.e. the neural network, it needs to be of form [#Inputs, [Size_Prediction_Path], [Size_Uncertainty_Path], #Outputs_Per_Path] e.g.: [3, [16, 8], [8, 4], 1]
    learning_rate: The learning rate for the Adam-Optimizer
    ensemble_size: The number of ensemble members, the number of outputs in each batch
    beta_pred: beta for L2-regularization of the weights in the prediction path
    beta_unc: beta for L2-regularization of the weights in the uncertainty path
    '''
    def __init__(self, nn_sizes, learning_rate,ensemble_size,beta_pred,beta_unc):
        self.ensemble_size = ensemble_size # number of ensemble members, i.e. number of outputs in each path
        self.size_pred = [nn_sizes[0]] + nn_sizes[1] + [nn_sizes[-1]] # size of layers in prediction path
        self.size_unc = [nn_sizes[0]] + nn_sizes[2] + [nn_sizes[-1]] # size of layers in uncertainty path
        self.input_data = tf.placeholder("float", [None, nn_sizes[0]], name='input_data')
        self.target_pred = tf.placeholder("float", [None, nn_sizes[-1]], name='target_pred')
        self.target_unc = tf.placeholder("float", [None, nn_sizes[-1]], name='target_unc')
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.Variable(learning_rate, trainable=False, name='learning_rate')
        # beta_*: L2-Norm factor for regularization
        self.beta_pred = beta_pred
        self.beta_unc = beta_unc
        '''
        Prediction path
        '''
        # Initialize the prediction path
        self.weights_pred = []
        self.reg_weights_pred = []
        self.biases_pred = []
        self.layer_pred = self.input_data
        for l in range(1,len(self.size_pred)-1):
            self.weights_pred.append(tf.Variable(tf.random_normal( [self.size_pred[l-1],self.size_pred[l]] ), name='weights_pred'))
            self.reg_weights_pred.append(tf.nn.l2_loss(self.weights_pred[l-1]))
            self.biases_pred.append(tf.Variable(tf.random_normal( [self.size_pred[l]] ), name='biases_pred'))
            self.layer_pred = tf.nn.tanh(tf.nn.xw_plus_b(self.layer_pred, self.weights_pred[l-1], self.biases_pred[l-1]))
        # Prediction path outputs
        self.weights_out_pred = []
        self.biases_out_pred = []
        self.output_pred = []
        for e in range(self.ensemble_size):
            self.weights_out_pred.append(tf.Variable(tf.random_normal( [self.size_pred[-2],self.size_pred[-1]] ), name='weights_out_pred_'+str(e)))
            self.biases_out_pred.append(tf.Variable(tf.random_normal( [self.size_pred[-1]] ), name='biases_pred_'+str(e)))
            self.output_pred.append(tf.matmul(self.layer_pred, self.weights_out_pred[e]) + self.biases_out_pred[e])
        '''
        Uncertainty path
        '''
        # Initialize the uncertainty path
        self.weights_unc = []
        self.reg_weights_unc = []
        self.biases_unc = []
        # first layer: both the Input data and the last hidden layer of the prediction path are inputs for this first layer uncertainty path layer
        self.layer_unc = tf.concat([self.layer_pred,self.input_data],1)
        self.weights_unc.append(tf.Variable(tf.random_normal( [self.size_pred[-2] + self.size_unc[0],self.size_unc[1]] ), name='weights_unc'))
        self.reg_weights_unc.append(tf.nn.l2_loss(self.weights_unc[0]))
        self.biases_unc.append(tf.Variable(tf.random_normal( [self.size_unc[1]] ), name='biases_unc'))
        self.layer_unc = tf.nn.tanh(tf.nn.xw_plus_b( self.layer_unc, self.weights_unc[0], self.biases_unc[0]))
        # following layers
        for l in range(2,len(self.size_unc)-1):
            self.weights_unc.append(tf.Variable(tf.random_normal( [self.size_unc[l-1],self.size_unc[l]] ), name='weights_unc'))
            self.reg_weights_unc.append(tf.nn.l2_loss(self.weights_unc[l-1]))
            self.biases_unc.append(tf.Variable(tf.random_normal( [self.size_unc[l]] ), name='biases_unc'))
            self.layer_unc = tf.nn.tanh(tf.nn.xw_plus_b( self.layer_unc, self.weights_unc[l-1], self.biases_unc[l-1]))
        # Uncertainty path outputs
        self.weights_out_unc = []
        self.reg_weights_out_unc = []
        self.biases_out_unc  = []
        self.output_unc = []
        for e in range(self.ensemble_size):
            self.weights_out_unc.append(tf.Variable(tf.random_normal( [self.size_unc[-2],self.size_unc[-1]] ), name='weights_out_unc_'+str(e)))
            self.biases_out_unc.append(tf.Variable(tf.random_normal( [self.size_unc[-1]] ), name='biases_unc_'+str(e)))      
            self.output_unc.append(tf.log( 1 +  tf.exp(tf.matmul(self.layer_unc, self.weights_out_unc[e]) + self.biases_out_unc[e]) ))
        '''
        Cost & Optimization settings
        '''
        self.cost_pred = []
        self.loss_unc = []
        self.cost_unc = []
        self.optimizer_pred = []
        self.optimizer_unc = []
        self.rw_pred_losses = []
        self.rw_unc_losses = []
        self.red_output_unc = tf.square(tf.subtract(self.output_pred,self.target_unc))
        self.delta_unc = tf.squared_difference(tf.reduce_mean(self.output_pred,0), self.target_unc)
        for e in range(self.ensemble_size):
            # cost prediction
            self.rw_pred_losses.append(self.beta_pred*self.reg_weights_pred[0])        
            for rw in range(1,len(self.reg_weights_pred)):
                self.rw_pred_losses[e] += self.beta_pred*self.reg_weights_pred[rw]
            self.cost_pred.append(tf.reduce_mean(tf.squared_difference(self.output_pred[e], self.target_pred)) + self.rw_pred_losses[e])
            # cost uncertainty
            self.loss_unc.append(tf.reduce_mean(tf.squared_difference(self.output_unc[e], self.delta_unc ) ) )   
            self.rw_unc_losses.append(self.beta_unc*self.reg_weights_unc[0])        
            for rw in range(1,len(self.reg_weights_unc)):
                self.rw_unc_losses[e] += self.beta_unc*self.reg_weights_unc[rw]                
            self.cost_unc.append(self.loss_unc[e] + self.rw_unc_losses[e])
            # optimizer 
            self.optimizer_pred.append(tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost_pred[e],global_step=self.global_step,
                var_list=[self.weights_pred,self.biases_pred,self.weights_out_pred[e],self.biases_out_pred[e]]))
            self.optimizer_unc.append(tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost_unc[e],global_step=self.global_step, 
                var_list=[self.weights_unc,self.biases_unc,self.weights_out_unc[e],self.biases_out_unc[e]]))
