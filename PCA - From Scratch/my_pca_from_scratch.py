import numpy as np

class My_PCA() :
    
    def run_pca(self,Covariance_matrix):
        
        S,V = self.find_eigens(Covariance_matrix)
        A_V = self.find_transformed_vectors(Covariance_matrix,V)
        U = self.gram_schmidt(A_V)
        
        return U,S,V
        
    def find_eigens(self,A,row_vect=False):
        
        if row_vect :
            A = A.T
            
        eigen_values, eigen_vectors = np.linalg.eig(A)
        sorted_indices = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[sorted_indices]
        eigen_vector_matrix = eigen_vectors[:, sorted_indices]
        
        return eigen_values, eigen_vector_matrix
        
    def find_transformed_vectors(self,A,B):
        return np.dot(A,B)

    def replace_zero(self,array): 
    
        for i in range(len(array)) :
            if array[i] == 0 : 
                array[i] = 1
        return array
    
    def gram_schmidt(self,A, norm=True, row_vect=False):
        """Orthonormalizes vectors by gram-schmidt process

        Parameters
        -----------
        A : ndarray,
        Matrix having vectors in its columns

        norm : bool,
        Do you need Normalized vectors?

        row_vect: bool,
        Does Matrix A has vectors in its rows?

        Returns
        -------
        G : ndarray,
        Matrix of orthogonal vectors

        """
        if row_vect :
            # if true, transpose it to make column vector matrix
            A = A.T

        no_of_vectors = A.shape[1]
        G = A[:,0:1].copy() # copy the first vector in matrix
        # 0:1 is done to to be consistent with dimensions - [[1,2,3]]

        # iterate from 2nd vector to number of vectors
        for i in range(1,no_of_vectors):

            # calculates weights(coefficents) for every vector in G
            numerator = A[:,i].dot(G)
            denominator = np.diag(np.dot(G.T,G)) #to get elements in diagonal
            weights = np.squeeze(numerator/denominator)

            # projected vector onto subspace G
            projected_vector = np.sum(weights * G,
                                      axis=1,
                                      keepdims=True)

            # orthogonal vector to subspace G
            orthogonalized_vector = A[:,i:i+1] - projected_vector

            # now add the orthogonal vector to our set
            G = np.hstack((G,orthogonalized_vector))

        if norm :
            # to get orthoNormal vectors (unit orthogonal vectors)
            # replace zero to 1 to deal with division by 0 if matrix has 0 vector
            G = G/self.replace_zero(np.linalg.norm(G,axis=0))

        if row_vect:
            return G.T

        return G
    
    def project_onto(self,X,U,K):
        Z = np.dot(X,U[:,:K]) # mxn - nxK
        return Z # mxK
    
    def recover_back(self,Z,U,K):
        X_rec = np.dot(Z,U[:,:K].T) # mxK - Kxn
        return X_rec # mxn
        
        
    
