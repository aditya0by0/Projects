import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class My_Logistic_Regression():
    
    def __init__(self):
        self.cost = None
        self.W = None
        self.b = None
        self.num_iters = 1000
        self.mu = {}
        self.std = {}
         
    def fit(self, X, Y, normalize = False , list_of_features=[], alpha=0.1, num_iters=1000, 
            lambda_=0.0):
        
        self.num_iters = num_iters
        
        if normalize:
            
            X_final,self.mu,self.std = self.feature_Normalize(X,
                                                             list_of_features,
                                                             self.mu,
                                                             self.std)
        
        else :
            
            X_final = X
        
        self.cost,self.W, self.b = self.gradient_descent(X_final.to_numpy(),
                                                         Y.to_numpy(),
                                                         alpha,
                                                         num_iters,
                                                         lambda_)
           
    def predict(self,X_input):
        
        if not self.mu and not self.std :
            
            X_norm = X_input
        
        else:
            
            X_norm,_,_ = self.feature_Normalize(X_input,
                                                [],
                                                self.mu,
                                                self.std)
            
        pred = self.sigmoid(np.dot(X_norm,self.W) + self.b)
        
        pred = pd.DataFrame(np.argmax(pred,axis=1),columns=['class'], index=X_input.index )
        
        pred = pd.get_dummies(pred,columns=['class'])
        
        pred = pred.rename(columns={'class_0' : 'Iris-setosa',
                                    'class_1' : 'Iris-versicolor',
                                    'class_2' : 'Iris-virginica'})
        
        return pred
    
    
    def feature_Normalize(self,X , list_of_features, p_mu , p_std):
        
        if len(list_of_features) == 0 : 
            list_of_features = X.columns
                
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
    
    def initialiaze_weights(self,X,Y):
        return np.zeros((X.shape[1],Y.shape[1]))
    
    def sigmoid(self,z) : 
        return 1/(1 + np.exp(-z))
    
    def compute_cost(self,X,Y,W,b,lambda_):
    
        m = Y.shape[0]

        
        Z = np.dot(X,W) + b
        A = self.sigmoid(Z) 
        J = (1/m) * np.sum( Y * - np.log(A) - (1 - Y) * np.log(1 - A)) + \
                    (lambda_/(2*m)) * np.sum(np.square(W)) #L2 regulazation
                       
        return J, A
    
    def gradient_descent(self,X,Y,alpha,num_iters,lambda_):
    
        m = Y.shape[0]
        W = self.initialiaze_weights(X,Y) 
        b = 1
        J_history = []

        for i in range(num_iters): 

            J, A = self.compute_cost(X,Y,W,b,lambda_)
            grad_W = (1/m) * np.dot(X.T,(A - Y)) + (lambda_/m) * W # Gradient with L2 Regularization
            grad_b = (1/m) * np.sum(A - Y) 

            W = W - alpha * grad_W
            b = b - alpha * grad_b

            J_history.append(J)

        return J_history,W,b
    
    def get_accuracy(self,Y_pred,Y_actual) : 
        
        accuracy = (np.sum(Y_pred == Y_actual) / Y_actual.size) * 100 
        accuracy = np.round(accuracy,2)
        return str(accuracy) + '%'
        
    def plot_cost(self) : 
        
        plt.figure(figsize=[5,4],dpi=100)
        plt.plot(list(range(self.num_iters)),self.cost)
        plt.title('Cost vs Number of Iterations')
        plt.xlabel('Number Of Iterations')
        plt.ylabel('Cost function')
        plt.show()
        
    def get_cost(self,Y_pred,Y_actual):
        
        m = Y_actual.shape[0]
        J = np.sum(np.square(Y_pred.values-Y_actual)).sum()/(2*m) 
        
        return J