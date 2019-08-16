from UncertaintyModel import UncertaintyModel
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

class ModelTrainer(object):
    '''
    Used to train a UncertaintyModel

    This works in two phases. 
    First, the prediction phase is optimized, where an ensemble is learning the underlying function f(x) = y. 
    The weights of the uncertainty part in the network are frozen during this phase.
    Second, the uncertainty phase is learning the relationship betweein x and the prediction error in the first phase. 
    Similarly, the weights of the prediction part in the network are frozen during this phase.

    Different learning configurations can be done, e.g.:
    
    learning-rate decay: 
    * learning_decay > 0: After every "update_step" epoch, the learning rate is multiplied by the learning_decay, e.g. learning_decay = 0.99 decreases the learning_rate by 1% every update_step
    * learning_decay < 0: After every "update_step" epoch, the batch_size increased relatively by abs(learning_decay), e.g. learning_decay = -1.1 increases the batch_size by 10% every update_step

    batch_size:
    * batch_size >= 1: Use absolute sample numbers as batch_size
    * batch_size > 0 < 1: use  batch_size relative to all avaailable training data, e.g. batch_size = 0.1 uses 10% of all the training_data in each batch pass
    * batch_size = -1: Uses all training data per batch pass

    update_step and max_stp_cnt:
    * update_step defines the interval of epochs when the current costs are displayed, the test-set cost is checked if it is still decreasing, and the learning_rate/batch_size are updated
    * max_stp_cnt: defines the number of update_steps that need to pass, where the test-set cost is increasing and, thus, the learning is stopped
    '''
    def __init__(self,nn_size,learning_rate,ensemble_size,beta_pred,beta_unc,random_seed):
        tf.set_random_seed(random_seed)
        np.random.seed(random_seed)
        self.nn_size = nn_size
        self.learning_rate = learning_rate
        self.ensemble_size = ensemble_size
        self.beta_pred = beta_pred
        self.beta_unc = beta_unc
        self.model = UncertaintyModel(self.nn_size,self.learning_rate,self.ensemble_size,self.beta_pred,self.beta_unc)
        self.random_seed = random_seed
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.model_trained = False

    def reset_graph(self):
        self.sess.run(tf.global_variables_initializer())    

    def train_model(self,X,y,epochs,batch_size,learning_decay,max_stop_cnt,update_step,continue_training=False):
        
        if continue_training == False:
            self.reset_graph()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=self.random_seed)
        boot_train_index = []
        BT_X_train = []
        BT_y_train = []
        boot_train = np.array(range(0,len(self.X_train)))
        batch_index = boot_train        
        batch_size_mult = abs(learning_decay)

        test_feed_pred = {self.model.input_data: self.X_test, self.model.target_pred: self.y_test}
        test_feed_unc = {self.model.input_data: self.X_test, self.model.target_unc: self.y_test}

        if batch_size >= 1: # absolute number of batch samples
            batch_runs = int(len(self.X_train)/batch_size)
        elif batch_size > 0 and batch_size < 1: # relative number of batch samples, i.e. 0.1 == 10% of total datasize
            batch_size = int(len(self.X_train)*batch_size)
            print 'New batch size: ', batch_size
            batch_runs = int(len(self.X_train)/batch_size)
        elif batch_size == -1: # use all
            batch_runs = 1

        init_batch_size = batch_size

        for b in range(2*self.ensemble_size):        
            boot_train_index.append(np.random.choice(boot_train, size=boot_train.shape, replace=True))
            BT_X_train.append(self.X_train[boot_train_index[b]])
            BT_y_train.append(self.y_train[boot_train_index[b]])

        # start training layer        
        print('Phase 1: Training prediction path! Uncertainty path remains frozen...')  
        early_stopping_counter = 0
        last_avg_test_cost = float("inf")
        last_avg_train_cost = float("inf")
        for itr in range(epochs):
            np.random.shuffle(batch_index)
            avg_cost = 0.0
            avg_test_cost = 0.0
            for cb in range(batch_runs):
                for b in range(self.ensemble_size):  
                    if batch_size != -1:
                        indices = batch_index[cb*batch_size:(cb+1)*batch_size-1]
                    else:
                        indices = range(BT_X_train[b].shape[0])
                    curX = BT_X_train[b][indices]
                    cury = BT_y_train[b][indices]      
                    feed = {self.model.input_data: curX, self.model.target_pred: cury}
                    _,c = self.sess.run([self.model.optimizer_pred[b],self.model.cost_pred[b]],feed_dict=feed)
                    avg_cost += c / batch_runs

            if itr % update_step == 0:                
                for b in range(self.ensemble_size):
                    tc = self.sess.run(self.model.cost_pred[b],feed_dict=test_feed_pred)
                    avg_test_cost += tc / batch_runs
                if avg_test_cost >= last_avg_test_cost:
                    early_stopping_counter += 1
                else:
                    early_stopping_counter = 0
                avg_train_cost_dec = (last_avg_train_cost - avg_cost)/last_avg_train_cost
                last_avg_train_cost = avg_cost 
                last_avg_test_cost = avg_test_cost
                if learning_decay > 0: 
                        self.sess.run(tf.assign(self.model.lr,self.model.lr*learning_decay))                
                elif batch_size <= len(self.X_train):
                    batch_size = int((batch_size_mult**itr)*init_batch_size)
                    if batch_size > len(self.X_train):
                        batch_size = len(self.X_train)
                    print 'New batch size: ',batch_size,' // Perc. of dataset: ', 100*batch_size/len(self.X_train)
                    batch_runs = int(len(self.X_train)/batch_size)
                print 'Iteration:', itr, avg_cost,  avg_test_cost, early_stopping_counter
            if early_stopping_counter >= max_stop_cnt:
                print 'Test error in Phase 1 not decreasing anymore... early stopping!'
                break
    
        print('Phase 2: Training Uncertainty path! Prediction path remains frozen...')  
        early_stopping_counter = 0
        last_avg_test_cost = float("inf")
        last_avg_train_cost = float("inf")
        self.sess.run(tf.assign(self.model.lr,self.learning_rate))
        batch_size = init_batch_size
        for itr in range(epochs):
            np.random.shuffle(batch_index)
            avg_cost = 0.0
            avg_test_cost = 0.0
            for cb in range(batch_runs):
                for b in range(self.ensemble_size):  
                    if batch_size != -1:
                        indices = batch_index[cb*batch_size:(cb+1)*batch_size-1]
                    else:
                        indices = range(BT_X_train[self.ensemble_size+b].shape[0])
                    curX = BT_X_train[self.ensemble_size+b][indices]
                    cury = BT_y_train[self.ensemble_size+b][indices]      
                    feed = {self.model.input_data: curX, self.model.target_unc: cury}
                    _,c = self.sess.run([self.model.optimizer_unc[b],self.model.cost_unc[b]],feed_dict=feed)
                    avg_cost += c / batch_runs

            if itr % update_step == 0:        
                for b in range(self.ensemble_size):
                    tc = self.sess.run(self.model.cost_unc[b],feed_dict=test_feed_unc)
                    avg_test_cost += tc / batch_runs
                if avg_test_cost >= last_avg_test_cost:
                    early_stopping_counter += 1
                else:
                    early_stopping_counter = 0
                avg_train_cost_dec = (last_avg_train_cost - avg_cost)/last_avg_train_cost
                last_avg_train_cost = avg_cost 
                last_avg_test_cost = avg_test_cost
                if learning_decay > 0: 
                        self.sess.run(tf.assign(self.model.lr,self.model.lr*learning_decay))                
                elif batch_size <= len(self.X_train):
                    batch_size = int((batch_size_mult**itr)*init_batch_size)
                    if batch_size > len(self.X_train):
                        batch_size = len(self.X_train)
                    print 'New batch size: ',batch_size,' // Perc. of dataset: ', 100*batch_size/len(self.X_train)
                    batch_runs = int(len(self.X_train)/batch_size)
                print 'Iteration:', itr, avg_cost,  avg_test_cost, early_stopping_counter
            if early_stopping_counter >= max_stop_cnt:
                print 'Test error in Phase 2 not decreasing anymore... early stopping!'
                break
        self.model_trained = True 
        
    def eval_model(self,X):
        pred_mat = np.array([]).reshape(X.shape[0],0)
        unc_mat = np.array([]).reshape(X.shape[0],0)
        feed = {self.model.input_data: X}
        for b in range(self.ensemble_size):
            pred,unc = self.sess.run([self.model.output_pred[b],  self.model.output_unc[b]],feed_dict=feed)
            pred_mat = np.concatenate((pred_mat,pred),axis=1)
            unc_mat = np.concatenate((unc_mat,unc),axis=1)

        return pred_mat, unc_mat

    def save_model(self, target_file):
        save_path = self.saver.save(self.sess, target_file)

    def close_session(self):
        self.sess.close()
    
    def get_train_data(self):
        if self.model_trained == True:
            return self.X_train, self.y_train
        else:
            return None, None

    def get_test_data(self):
        if self.model_trained == True:
            return self.X_test, self.y_test
        else:
            return None, None