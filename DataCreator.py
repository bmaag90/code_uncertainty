import numpy as np

class DataCreator(object):
    '''
    Creates random datapoints (x,y) of the underlying function
        y = f + N(0,noise^2)
    with 
        f = sin(5/2*x)*son(3/2*x)
    and
        noise = 0.15 + 0.25*(1-sin(5/2*x))^2

    '''
    def __init__(self,num_samples, x_min=0, x_max=2*np.pi,random_seed=42):
        np.random.seed(random_seed)
        self.num_samples = num_samples
        self.x_min = x_min
        self.x_max = x_max
    
    '''
    call to sample new data points
    '''
    def create_datapoints(self):
        self.x = np.linspace(self.x_min,self.x_max,num=self.num_samples)
        self.f = np.sin(5*self.x/2)*np.sin(3*self.x/2)
        self.noise = 0.15 + 0.25*(1-np.sin(5*self.x/2))**2
        self.y = self.f+np.random.randn(self.num_samples)*self.noise

        self.f = self.f.reshape(self.num_samples,1)
        self.noise = self.noise.reshape(self.num_samples,1)
        self.x = self.x.reshape(self.num_samples,1)
        self.y = self.y.reshape(self.num_samples,1)

    def get_data(self):
        return self.x, self.y, self.f, self.noise

    '''
    get normalized samples
    '''
    def get_normalized_data(self):
        mux = np.mean(self.x)
        sigx = np.std(self.x)
        muy = np.mean(self.y)
        sigy = np.std(self.y)

        return (self.x-mux)/sigx, (self.y-muy)/sigy, mux, sigx, muy, sigy
