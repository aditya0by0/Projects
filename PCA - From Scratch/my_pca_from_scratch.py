import numpy as np
import matplotlib.pyplot as plt

class My_PCA() :
    
    def __init__(self):
        
        # object variable to store number of first 'K' Principal Components (PCs) to 
        # which we are projecting data on, so as to recover back data from same PCs
        self.K = None
        
        self.exp_var = [] # stores explained variances by each PCs
    
    def run_pca(self,norm_X):
        """ Computes Covariance Matrix and calculates eigen vectors for
        the our given Data Matrix.
        
        Parameters
        ----------
        Data_Matrix : ndarray,
            Normalized Matrix containing examples on its columns and 
            features on its rows - shape mxn
            
        Returns
        -------
        eig_vect : ndarray,
            Eigen vectors for our Covariance Matrix
        """
        
        Cov = self.cal_cov(norm_X) # Covariance Matrix
        eig_val, eig_vect = self.find_eigens(Cov) # Eigen vectors and values
        self.exp_var = self.cal_var_pc(eig_val) # Explained Variance by each PCs
        
        return eig_vect
    
    def cal_cov(self,X):
        """ Calculates Covariance matrix for our data matrix 'X'
        which have examples on its rows and features on its columns.
        
        Parameters
        ----------
        X : ndarray,
            Normalized Matrix containing examples on its columns and 
            features on its rows - shape mxn
            
        Returns
        -------
        Cov : ndarray,
            Matrix which have variances of our features on diagonal elements
            and covariances between our features on non-diagonal elements
            shape - nxn
            
        """
        
        m = X.shape[0] # no. of examples 
        Cov = (1/m) * np.dot(X.T,X) # nxm - mxn = nxn
        
        return Cov
        
    def find_eigens(self,Cov):
        """ Calculates eigen vectors and eignen values values for
        covariance matrix and sorts the eigen values and its 
        corresponding eigen vectors in decreasing order.
        
        Parameters
        ----------
        Covariance_matrix : ndarray,
            Matrix which have variances of our features on diagonal elements
            and covariances between our features on non-diagonal elements
            shape - nxn
        
        Returns
        -------
        eigen_values : numpy array,
            Array having eigen values in decreasing order
        
        eigen_vectors : ndarray,
            Matrix having eigen vectors in its columns sorted as per
            eigen values, these vector are also called Principal Components
            
        """
        
        # returns tuple of eigen values and unit eigen vectors
        eig_val, eig_vect = np.linalg.eig(Cov)
        
        # sort eigen values in decreasing order and capture indices
        sort_indices = np.argsort(eig_val)[::-1] 
        
        # appling the indices to eigen values to sort it by
        eig_val = eig_val[sort_indices] 
        
        # appling the indices to eigen values to sort it in same order
        eig_vect = eig_vect[:,sort_indices] 
        
        return eig_val, eig_vect
        
    def cal_var_pc(self,eig_val):
        """ Calculates the percentage of variance explained or captured
        by each principal component in its direction.
        
        Parameters
        ----------
        eig_val : numpy array,
            numpy array of eigen values
           
        Returns
        -------
        exp_var : python list,
            list containing percentage of variance explained or captured 
            by each principal component
        
        """
        
        total_variance = np.sum(eig_val) # total variance
        exp_var = [] # percentage of variance explained by each principal component
        
        # iterating over eigen values
        for i in eig_val: 
            
            # percentage of variance captured by this principal component (Pc)
            var_this_pc = round(i/total_variance,2) * 100
            
            exp_var.append(var_this_pc) # append to our list

        return exp_var
        
    
    def project_onto(self,X,U,K):
        """ Projects our data on first 'K' principal components
        
        Parameters
        ----------
        X : ndarray,
            Our Data Matrix - shape mxn
        
        U : ndarray,
            Eigen vectors Matrix or Principal Components on which
            we will be projecting our data - shape nxn
        
        K : int,
            Number of first Principal Components on to which we want 
            to project our data
            
        Returns
        -------
        Z : ndarray,
            Projected data on first 'K' Principal Components 
            - shape mxK
        """
        
        self.K = K # remember K, so as to recover back from same 'K' PCs
        Z = np.dot(X, U[:, : self.K]) # mxn - nxK = mxK
        
        return Z
    
    def recover_back(self,Z,U):
        """ Recovers back our projected data to orginal dimensional space
        
        Parameters
        ----------
        Z : ndarray,
            Our Projected Data - shape mxK
            
        U : ndarray,
            Eigen vectors Matrix or Principal Components on which we have 
            projected our data - shape nxn
           
        Returns
        -------
        X_rec : ndarray,
            Our Recovered data to orginal dimension - shape mxn
        """
        
        X_rec = np.dot(Z,U[:, : self.K].T) # mxK - Kxn = mxn
        
        return X_rec
    
    def plot_exp_var(self):
        """ Plots variance explained by each Principal Component
        """
        
        name = [] # for PCs naming
        
        for i in range(len(self.exp_var)):
            name.append('PC' + str(i+1))
        
        plt.figure()
        plt.bar(name,self.exp_var) # bar graph
        plt.xlabel('Principal Components')
        plt.ylabel('Variance explained by each PC')
        plt.show()
        
        
        
    
