import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class My_Linear_Regression():

    def __init__(self):
        """ Initializes object variables
        """

        self.cost = None # stores training cost array
        self.theta = [] # theta our data set
        self.num_iters = 100 # number of Iterations
        self.X_final = pd.DataFrame() # stores our normalized data set
        self.Y = pd.Series(dtype='int64') # stores target data set
        self.mu = {} # dictionary of means
        self.std = {} # dictionary of standard deviations
        self.train_r2_score = None # R_square score

    def fit(self, X, Y, normalize = False , list_of_features=[], alpha=0.1, num_iters=100,
            lambda_=0.0):
        """ This method first normalizes the data based on boolean parameter
        then optimizes the theta by performing gradient descent after adding
        intercept to data and then Calculates R_square score for the same.

        Parameters
        ----------
        X : DataFrame,
            DataFrame with features (Data Matrix)

        Y : Series,
            Series with actual values (target values)

        normalize : bool,
                boolean to tell whether to normalize data or not.

        list_of_features : python list,
            List containing names of the columns to normalize

        alpha : float or int,
            learning rate for gradient descent

        num_iters : int,
            number of Iterations the gradient descent should return

        lambda_ : float,
            regularization parameter used to prevent overfitting

        Returns
        -------
        None
        """
        # store parameters in our object variables
        self.num_iters = num_iters
        self.Y = Y

        # Whether to normalize the data or not
        if normalize:

            X_norm,self.mu,self.std = self.feature_Normalize(X,
                                                             list_of_features,
                                                             self.mu,
                                                             self.std)
            # store to means and standard deviations to normalize our
            # dev and test data by same means and standard deviations

        else :

            X_norm = X

        # Adding intercept column (bias term) to our data set
        X_norm_intercept = self.add_intercept(X_norm)

        # Storing the transfomed data set in our object variable
        self.X_final = X_norm_intercept

        # Initialize theta to zeros
        self.initialize_theta(X_norm_intercept)

        # perfom gradient descent
        self.theta, self.cost = self.gradient_descent(self.X_final,
                                                      self.Y,
                                                      self.theta,
                                                      alpha,
                                                      num_iters,
                                                      lambda_)

        # Calculate R_square score for our dataset
        self.train_r2_score = self.calculate_r2_score(self.Y,
                                                      np.dot(self.X_final,self.theta),
                                                      np.mean(self.X_final,axis=1))

    def predict(self,X_input):
        """ Predict the output (target) based on optimized theta.

        Parameters
        ----------
        X_input : DataFrame,
            DataFrame with features (Data Matrix)

        Returns
        -------
        numpy array with predicted values

        """

        # Check if our training data was normalized or not
        if not self.mu and not self.std :

        # if here, training data was not normalized, so just add intercept
            X_intercept = self.add_intercept(X_input)

            # predict based on our theta
            return np.dot(X_input,self.theta)

        else:

        # As our training data was normalize, so use training data's
        # means and standard deviation to normalize this data
            X_norm,_,_ = self.feature_Normalize(X_input,
                                                [],
                                                self.mu,
                                                self.std)
            # now add intercept
            X_intercept_norm = self.add_intercept(X_norm)

            # predict based on our theta
            return np.dot(X_intercept_norm,self.theta)


    def feature_Normalize(self,X , list_of_features, p_mu , p_std):
        """ Normalizes the columns specified by 'list_of_features' parameter
        by subtracting the values by mean of respective column and then
        scaling its values by standard deviation of that column.

        Parameters
        ----------
        X : DataFrame,
            DataFrame with features (Data Matrix)

        list_of_features : python list,
            List containing names of the columns to normalize

        p_mu : numpy array,
            array of means by which to subract the values of the respective
            columns

        p_std : numpy array,
            array of standard deviation by which to scale the values of the
            respective columns

        Returns
        -------
        None
        """
        # if users doesnt give which columns (features) to normalize,
        # used this default features based on our dataset
        if len(list_of_features) == 0 :
            list_of_features = ['Present_Price(lacs)','Kms_Driven','Age']

        X_norm = X.copy()
        mu_dict = {}
        std_dict = {}

        # loop and normalize each feature
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
            std_dict[i] = std # stores standard deviations

        return X_norm, mu_dict, std_dict


    def add_intercept(self,X):
        """ Adds intercept term (bias term) in new column named
        intercept in the given dataframe 'X' and places that column
        in first position of DataFrame.

        Parameters
        ----------
        X : DataFrame,
            DataFrame with features (Data Matrix)

        Returns
        -------
        X_intercept : DataFrame,
            DataFrame with intercept column.
        """

        X['intercept'] =  np.ones(X.shape[0]) # add intercept (bias)
        # Make this as first column of our dataset
        X_intercept = X.reindex(columns= (['intercept'] + X.columns.to_list()[:-1]))

        return X_intercept


    def initialize_theta(self,X):
        """ Initializes theta to array of zeros with shape equal
        to number of features in given X and stores the theta in
        object variable (self.theta).

        Parameters
        ----------
        X : DataFrame,
            DataFrame with features (Data Matrix)

        Returns
        -------
        None
        """

        # Initialize as per number of features in our dataset and
        # store it in our object variable
        self.theta = np.zeros(X.shape[1])



    def compute_cost(self,X,Y,theta,lambda_):
        """ Calculates the Cost with help of Cost function for Linear
        Regression for the given Parameters.

        Parameters
        ----------
        X : DataFrame,
            DataFrame with features (Data Matrix)

        Y : Series,
            Series with actual values (target values)

        theta : numpy array,
            theta or weights to compute cost

        lambda_ : float,
            regularization parameter used to prevent overfitting

        Returns
        -------
        J : float,
            returns cost of the function
        """

        m = Y.shape[0] # number of training examples
        h = np.dot(X,theta) # compute hypothesis based on theta

        # Compute Cost as per Cost function
        J = np.sum(np.square(h-Y))/(2*m) + \
            (lambda_/(2*m)) * np.sum(np.square(theta[1:])) # l2 regularization

        return J

    def gradient_descent(self,X,Y,theta,alpha,num_iters,lambda_):
        """ Calculates the gradient of cost function and perfoms
        gradien descent as per number of Iterations given.

        Parameters
        ----------
        X : DataFrame,
            DataFrame with features (Data Matrix)

        Y : Series,
            Series with actual values (target values)

        theta : numpy array,
            initial theta to start gradient descent

        alpha : float or int,
            learning rate for gradient descent

        num_iters : int,
            number of Iterations the gradient descent should return

        lambda_ : float,
            regularization parameter used to prevent overfitting

        Returns
        -------
        theta  : numpy array,
            optimized theta after doing performing gradient gradient_descent

        J_history : python list,
            List containing cost computed in every Iteration.
        """
        m = Y.shape[0] # number of training examples
        theta = theta.copy() # copy the theta to new variable
        J_history = [] # list to record cost in each Iteration
        grad = np.zeros(theta.shape) # to make sure grad is of same shape

        for i in range(num_iters):

            grad = (1/m) * ( (np.dot(X,theta) - Y).dot(X) )

            # Update theta as per gradient Calculated
            theta[0] = theta[0] - alpha * grad[0] # We dont regularize intercept term
            theta[1:] = theta[1:] - alpha * grad[1:] + (lambda_/m) * theta[1:]

            # append cost of this Iteration
            J_history.append(self.compute_cost(X,Y,theta,lambda_))

        return theta, J_history

    def calculate_r2_score(self,Y_actual,Y_pred,mean):
        """ Calculates the R_square score for the given Parameters.

        Parameters
        ----------
        Y_actual : DataFrame,
            DataFrame which has actual values for certain set

        Y_pred : DataFrame,
            DataFrame which predicted values for the above set

        mean : panda series or numpy array,
            shape must be equal to Y_actual
            Mean of the training examples across features

        Returns
        -------
        R_square  : float
            R_square score
        """

        numerator  = np.sum(np.square(Y_actual - Y_pred)) # numerator
        denominator = np.sum(np.square(Y_actual - mean)) # denominator
        R_square = 1 - numerator/denominator # R square score

        return R_square

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
        plt.figure() # new figure
        plt.plot(list(range(self.num_iters)),self.cost)
        plt.title('Cost Vs Number of Iterations')
        plt.xlabel('Number Of Iterations')
        plt.ylabel('Cost function')
        plt.show() # show the plot

    def plot_graph_for_r2(self) :
        """ Plots graph for showing relationship between predicted
        and actual values for the training set.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        plt.figure() # new figure
        plt.scatter(np.dot(self.X_final,self.theta), self.Y)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show() # show plot

    def get_cost(self,Y_pred,Y_actual):
        """ Computes and returns the cost of the function.

        Parameters
        ----------
        Y_pred : DataFrame,
            DataFrame which gives prediction for certain set

        Y_actual : DataFrame,
            DataFrame which actual values for the above set

        Returns
        -------
        J   : float
            cost of the Parameters
        """
        m = Y_actual.shape[0] # number of training examples
        J = np.sum(np.square(Y_pred-Y_actual))/(2*m) # cost

        return J
