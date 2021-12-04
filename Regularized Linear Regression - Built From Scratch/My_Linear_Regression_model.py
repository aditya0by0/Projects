import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class My_Linear_Regression():

    def __init__(self):
        
        self.cost = None
        self.theta = []
        self.num_iters = 100
        self.X_final = pd.DataFrame()
        self.Y = pd.Series(dtype='int64')
        self.mu = {}
        self.std = {}
        self.train_r2_score = None
        
    def fit(self, X, Y, normalize = False , list_of_features=[], alpha=0.1, num_iters=100, 
            lambda_=0.0):
        
        self.num_iters = num_iters
        self.Y = Y
        
        if normalize:
            
            X_norm,self.mu,self.std = self.feature_Normalize(X,
                                                             list_of_features,
                                                             self.mu,
                                                             self.std)
        
        else :
            
            X_norm = X 
            
        X_norm_intercept = self.add_intercept(X_norm)
        
        self.X_final = X_norm_intercept
        
        self.initialize_theta(X_norm_intercept)
        
        self.theta, self.cost = self.gradient_descent(self.X_final,
                                                      self.Y,
                                                      self.theta,
                                                      alpha,
                                                      num_iters,
                                                      lambda_)
        
        self.train_r2_score = self.calculate_r2_score(self.Y,
                                                      np.dot(self.X_final,self.theta), 
                                                      np.mean(self.X_final,axis=1) )
      
    def predict(self,X_input):
        
        if not self.mu and not self.std :
            
            X_intercept = self.add_intercept(X_input)
            
            return np.dot(X_input,self.theta)
        
        else:
            
            X_norm,_,_ = self.feature_Normalize(X_input,
                                                [],
                                                self.mu,
                                                self.std)
            
            X_intercept_norm = self.add_intercept(X_norm)
            
            return np.dot(X_intercept_norm,self.theta)
    
    
    def feature_Normalize(self,X , list_of_features, p_mu , p_std):
        
        if len(list_of_features) == 0 : 
            list_of_features = ['Present_Price(lacs)','Kms_Driven','Age']
                
        X_norm = X.copy()
        mu_dict = {}
        std_dict = {}

        for i in list_of_features : 
            
            if i in self.mu.values() and i in self.std.values() :
                
                mu = p_mu[i]
                std = p_std[i]
            
            else : 
                
                mu = np.mean(X_norm[i])
                std = np.std(X_norm[i])
            
            X_norm[i] = (X_norm[i] - mu)/std
            mu_dict[i] = mu
            std_dict[i] = std

        return X_norm, mu_dict, std_dict

    
    def add_intercept(self,X):
        
        X['intercept'] =  np.ones(X.shape[0])
        X_intercept = X.reindex(columns= (['intercept'] + X.columns.to_list()[:-1]))
        
        return X_intercept
        
    
    def initialize_theta(self,X):
        
        self.theta = np.zeros(X.shape[1])
        
        return 0 

    
    def compute_cost(self,X,Y,theta,lambda_):
    
        m = Y.shape[0]
        J = 0
        h = np.dot(X,theta)
        J = np.sum(np.square(h-Y))/(2*m) + (lambda_/(2*m)) * np.sum(np.square(theta[1:]))
    
        return J
    
    def gradient_descent(self,X,Y,theta,alpha,num_iters,lambda_):
    
        m = Y.shape[0]
        theta = theta.copy()
        J_history = []
        grad = np.zeros(theta.shape)
    
        for i in range(num_iters):

            grad = (alpha/m) * ( (np.dot(X,theta) - Y).dot(X) )
            theta[0] = theta[0] - grad[0]
            theta[1:] = theta[1:] - grad[1:] + (lambda_/m) * theta[1:]

            J_history.append(self.compute_cost(X,Y,theta,lambda_))

        return theta, J_history
    
    def calculate_r2_score(self,Y_actual,Y_pred,mean):
    
        return 1 - (np.sum(np.square(Y_actual - Y_pred))/(np.sum(np.square(Y_actual - mean))))
    
    def plot_cost(self) : 
        
        plt.figure()
        plt.plot(list(range(self.num_iters)),self.cost)
        plt.xlabel('Number Of Iterations')
        plt.ylabel('Cost function')
        plt.show()
        
    def plot_graph_for_r2(self) :
        
        plt.figure()
        plt.scatter(np.dot(self.X_final,self.theta), self.Y)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    
    def get_cost(self,Y_pred,Y_actual):
        
        m = Y_actual.shape[0]
        J = np.sum(np.square(Y_pred-Y_actual))/(2*m) 
        
        return J