import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class My_Logistic_Regression():

    def __init__(self):
        """ Initializes object variables
        """
        self.cost = None # cost in each Iteration
        self.W = None # weights for each feature
        self.b = None # bias term (bias weight)
        self.num_iters = None # default number of Iterations
        self.mu = {} # dictionary of means
        self.std = {} # dictionary of standard deviations
        self.accuracy = None # training accuracy

    def fit(self, X, Y, normalize = False , list_of_features=[], alpha=0.1,
            num_iters=1000, lambda_=0.0):
        """ This method first normalizes the data based on boolean parameter
        then optimizes the weights and bias by performing gradient descent
        and then calculates accuracy for the same.

        Parameters
        ----------
        X : DataFrame,
            DataFrame containing features (Data Matrix)

        Y : DataFrame,
            DataFrame with actual values (target values)
            - one hot encoded

        normalize : bool,
                boolean to tell whether to normalize data or not.

        list_of_features : python list,
            List containing names of the columns to normalize

        alpha : float or int,
            learning rate for gradient descent

        num_iters : int,
            number of Iterations the gradient descent should run

        lambda_ : float,
            regularization parameter used to prevent overfitting

        Returns
        -------
        None
        """
        self.num_iters = num_iters # store in object variable

        # Whether to normalize data or not
        if normalize:

            X_final,self.mu,self.std = self.feature_Normalize(X,
                                                             list_of_features,
                                                             self.mu,
                                                             self.std)
            # store the means and standard deviations to normalize our
            # dev and test data by same means and standard deviations
        else :

            X_final = X

        # perform gradient descent
        self.cost,self.W, self.b = self.gradient_descent(X_final.to_numpy(),
                                                         Y.to_numpy(),
                                                         alpha,
                                                         num_iters,
                                                         lambda_)

        # Calculate accuracy for training set
        Y_pred = self.predict(X_final) # predict based on optimized weights
        Y_pred = Y_pred.idxmax(axis=1) # column name of max values
        self.accuracy = self.get_accuracy(Y_pred,
                                          Y.idxmax(axis=1))

    def predict(self,X_input):
        """ Predict the output (target) based on optimized weights
        and bias.

        Parameters
        ----------
        X_input : DataFrame,
            DataFrame containing features (Data Matrix)

        Returns
        -------
        pred : DataFrame,
            DataFrame with one hot encoding for output

        """
        # Check if our training data was normalized or not
        if not self.mu and not self.std :
            # if here, training data was not normalized
            X_final = X_input

        else:
            # if here, training data was normalized, so normalize
            # the X_input based on means and standard deviation of
            # training data
            X_final,_,_ = self.feature_Normalize(X_input,
                                                [],
                                                self.mu,
                                                self.std)

        # predict based on our optimized weights and bias
        pred = self.sigmoid(np.dot(X_final,self.W) + self.b)

        # get the index of highest values and create dataframe based on
        # same index of X_input
        pred = pd.DataFrame(np.argmax(pred,axis=1),columns=['class'],
                            index=X_input.index )

        # one hot encoding
        pred = pd.get_dummies(pred,columns=['class'])

        # change column names
        pred = pred.rename(columns={'class_0' : 'Iris-setosa',
                                    'class_1' : 'Iris-versicolor',
                                    'class_2' : 'Iris-virginica'})

        return pred

    def feature_Normalize(self,X , list_of_features, p_mu , p_std):
        """ Normalizes the columns specified by 'list_of_features' parameter
        by subtracting the values by mean of respective column and then
        scaling its values by standard deviation of that column.

        Parameters
        ----------
        X : DataFrame,
            DataFrame containing features (Data Matrix)

        list_of_features : python list,
            List containing names of the columns to normalize

        p_mu : dictionary,
            dictionary of means by which to subract the values of the
            respective columns

        p_std : dictionary,
            dictionary of standard deviations by which to scale the values
            of the respective columns

        Returns
        -------
        X_norm : DataFrame,
            normalized dataframe for given X

        mu_dict : dictionary,
            dictionary of means

        std_dict : dictionary,
            dictionary of standard deviations
        """
        # if users doesnt give which columns (features) to normalize,
        # normalize all the columns
        if len(list_of_features) == 0 :
            list_of_features = X.columns

        X_norm = X.copy()
        mu_dict = {}
        std_dict = {}

        # loop and normalize each column
        for i in list_of_features :

            if i in self.mu.values() and i in self.std.values() :
            # if we already have means and standard deviations values of
            # our training set
                mu = p_mu[i]
                std = p_std[i]

            else :
            # Calculate the means and standard deviation and then normalize
                mu = np.mean(X_norm[i])
                std = np.std(X_norm[i])

            X_norm[i] = (X_norm[i] - mu)/std # normalize
            mu_dict[i] = mu # store means
            std_dict[i] = std # store standard deviations

        return X_norm, mu_dict, std_dict

    def initialiaze_weights_bias(self,X,Y):
        """ Initializes weights to numpy array of zeros with shape equal
        to number of features by number of classes and bias to shape equal
        to one by number of classes.

        Parameters
        ----------
        X : DataFrame,
            DataFrame containing features (Data Matrix)

        Y : DataFrame,
            DataFrame of target variable - one hot encoded

        Returns
        -------
        W : numpy array,
            weights initialized to zeros (shape = #features x #classes)

        b : bias,
            bias initialized to zeros (shape = 1 x #classes)
        """
        # no. of featurs x no. of classes - Initialize to array of zeros
        W = np.zeros((X.shape[1],Y.shape[1]))

        # 1 x no.of classes - Initialize to array of zeros
        # bias term for each class
        b = np.zeros((1,Y.shape[1]))
        return W, b

    def sigmoid(self,z) :
        """ Applies sigmoid activation to input array or dataframe

        Parameters
        ----------
        z : numpy array or dataframe

        Returns
        -------
        numpy array or dataframe with activation function applied
        """

        return 1/(1 + np.exp(-z)) # activation function

    def compute_cost(self,X,Y,W,b,lambda_):
        """ Calculates the Cost with help of Cost function for Logistic
        Regression for the given Parameters.

        Parameters
        ----------
        X : numpy array,
            array with features (Data Matrix)

        Y : numpy array,
            array with actual values (target values) - one hot encoded

        W : numpy array,
            weights to compute cost

        b : numpy array,
            bias term or bias weight

        lambda_ : float,
            regularization parameter used to prevent overfitting

        Returns
        -------
        J : float,
            cost of the function

        A : numpy array,
            predicted values
        """
        m = Y.shape[0] # number of training examples

        Z = np.dot(X,W) + b # linear step
        A = self.sigmoid(Z) # adding non-linearity (activation function)

        # cost function for Logistic Regression with regulazation
        J = (1/m) * np.sum( Y * - np.log(A) - (1 - Y) * np.log(1 - A)) + \
                    (lambda_/(2*m)) * np.sum(np.square(W)) #L2 regulazation

        return J, A

    def gradient_descent(self,X,Y,alpha,num_iters,lambda_):
        """ Calculates the gradient of cost function and perfoms
        gradient descent number of times given by num_iters.

        Parameters
        ----------
        X : numpy array,
            array with features (Data Matrix)

        Y : numpy array,
            array with actual values (target values) - one hot encoded

        alpha : float or int,
            learning rate for gradient descent

        num_iters : int,
            number of Iterations the gradient descent should run

        lambda_ : float,
            regularization parameter used to prevent overfitting

        Returns
        -------
        J_history : python list,
            List containing cost computed in every Iteration.

        W : numpy array,
            array of optimized weights

        b : numpy array,
            optimized bias term or bias weight
        """

        m = Y.shape[0] # number of training examples
        W, b = self.initialiaze_weights_bias(X,Y) # Initialize weights and bias
        J_history = [] # list to record cost in each Iteration

        for i in range(num_iters):

            J, A = self.compute_cost(X,Y,W,b,lambda_) # compute cost

            # gradient for weights
            grad_W = (1/m) * np.dot(X.T,(A - Y)) + \
                     (lambda_/m) * W # L2 Regularization
            # gradient for bias terms
            grad_b = (1/m) * np.sum(A - Y, axis=0, keepdims=True)

            W = W - alpha * grad_W # gradient descent
            b = b - alpha * grad_b

            J_history.append(J) # record cost for each Iteration

        return J_history,W,b

    def get_accuracy(self,Y_pred,Y_actual) :
        """ Calculate accuracy of our predictions

        Parameters
        ----------
        Y_pred : Series,
            Series of predicted values

        Y_actual : Series,
            Series of Actual values

        Returns
        -------
        Accuracy in string format
        """
        # Compute accuracy
        accuracy = (np.sum(Y_pred == Y_actual) / Y_actual.size) * 100
        accuracy = np.round(accuracy,2) # round to two decimals
        return str(accuracy) + '%' # add % symbol and return as string

    def plot_cost(self) :
        """ Plots graph for Change in cost function as the number of
        Iterations increases for training set.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        plt.figure(figsize=[5,4],dpi=100) # new figure
        plt.plot(list(range(self.num_iters)),self.cost)
        plt.title('Cost vs Number of Iterations')
        plt.xlabel('Number Of Iterations')
        plt.ylabel('Cost function')
        plt.show() # show the plot

    def get_linear_cost(self,Y_pred,Y_actual):
        """ Computes and returns the cost of the function.

        Parameters
        ----------
        Y_pred : DataFrame,
            DataFrame which gives prediction for certain set
            - one hot encoded

        Y_actual : DataFrame,
            DataFrame which actual values for the above set
            - one hot encoded

        Returns
        -------
        J   : float
            cost of the Parameters
        """

        m = Y_actual.shape[0] # number of training examples
        # cost of Calculated by subtracting and averaging over
        J = np.sum(np.square(Y_pred.values-Y_actual)).sum()/(2*m)

        return J
