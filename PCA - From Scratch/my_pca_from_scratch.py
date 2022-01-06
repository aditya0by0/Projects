import numpy as np

class My_PCA() :
    
    self.K = None
    
    def run_pca(self,Covariance_matrix):
        
        S,V = self.find_eigens(Covariance_matrix)
        A_V = self.find_transformed_vectors(Covariance_matrix,V)
        U = self.gram_schmidt(A_V)
        
        return U,S,V
        
    def find_eigens(self,A,row_vect=False):
        
        if row_vect :
            A = A.T
        
        # returns tuple of eigen values and unit eigen vectors
        eigen_values, eigen_vectors = np.linalg.eig(Cov_matrix) 
        sort_indices = np.argsort(eigen_values)[::-1] # sort the indices of  eigen value from large to small
        eigen_values = eigen_values[sort_indices] # sort the eigen values
        eigen_vectors = eigen_vectors[:,sort_indices] # sort eigen column vectors in same order
        
        return eigen_values, eigen_vectors
        
    def cal_var_pc(array):
    
        total_variance = np.sum(array)
        var_in_pc = []
        for i in array:
            var_4_pc = round(i/total_variance,2)
            var_in_pc.append(var_4_pc)

        return var_in_pc
        
    
    def project_onto(self,X,U,K):
        
        self.K = K
        Z = np.dot(X, U[:, : self.K]) # mxn - nxK = mxK
        
        return Z
    
    def recover_back(self,Z,U):
        
        X_rec = np.dot(Z,U[:, : self.K].T) # mxK - Kxn = mxn
        
        return X_rec
        
        
    
