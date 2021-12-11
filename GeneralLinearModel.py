## The purpose of this module is to demonstrate knowledge of generalized linear models and maximum liklihood estimation by 
## creating a module that will be able to fit a data set to a chosen general linear model and produce parameters beta. 
## Also because I'm a nerd and this is fun to me.
## Python was chosen due to familiarity, but realistically, because this method relies on performing many base level computations, 
## it would be written in C/C++ in a real world sitionation.  Most statistical packages already have this functionality implimented anyways, this is simply for
## learning sake. 

import pandas as pd #for handling data
import numpy as np #for doing matrix math

class GeneralLinearModel:
    '''
    This class will take data as a pandas data frame and model type as a string to produce a general linear model.  This class will automatically perform maximum 
    liklihood estimation and store the optimized parameters as attributes along with calculating the goodness of fit statistic and performing a wald test. 

    '''

    def __init__(self, data, model_type, target_variable, param_init = None, convergence = .001, max_itter = 100):
        self.param_init = param_init
        self.target = target_variable
        self.target_vector = data[target_variable].to_numpy()
        self.design_matrix = data.drop(columns = [target_variable]).to_numpy()
        self.model_type = model_type
        self.mean_function_derivative = self.setMeanFunctionDerivative(target_variable)
        self.mean_fucntion = self.setMeanFunction(model_type)
        self.num_param = self.design_matrix.shape[1]
        self.num_data = data.shape[0]
        self.max_itter = max_itter
        self.convergence_limit = convergence
        self.parameters = self.itterativeLeastSquares()

    def setMeanFunctionDerivative(self, model_type):
        if(self.model_type == 'linear'):
            def derivative(eta):
                return 1
        elif(self.model_type == 'logistic'):
            def derivative(eta):
                return ( (np.exp(eta) ) / ( (1 + np.exp(eta))**2 ) )
        return derivative

    def setMeanFunction(self, model_type):
        if(self.model_type == 'linear'):
            def mean_function(current_parameters, data_row):
                return np.dot(current_parameters, data_row)

        if self.model_type == 'logistic':
            def mean_function(current_parameters, data_row):
                return( 1 / 1 + np.exp(-np.dot(current_parameters, data_row)))
        return mean_function

    def targetVariance(self, current_parameters):
        estimated = np.matmul(self.design_matrix, current_parameters)

        deviance = self.target_vector - estimated
        

        deviance_dot = np.dot(deviance.transpose(), deviance)
    
        variance = (1 / (self.num_data - self.num_param)) * deviance_dot

        return variance


    def calcMeanFunctionDerivative(self, current_parameters, data_row):
        eta = np.dot(current_parameters, data_row)
        mean_function_derivative = self.mean_function_derivative(eta)
        return mean_function_derivative

    def calcW(self, current_parameters):
        '''

        J = X_transpose * W * X

        where W _ii = 1 / V(y) * (dmu/deta)

        This method calculates the matrix W
        '''
        variance = self.targetVariance(current_parameters)

        W = np.zeros((self.num_data, self.num_data))

        for i in range(0, self.num_data):
            W[i][i] = (1 / variance) * self.calcMeanFunctionDerivative(current_parameters, self.design_matrix[i])
            
        return W

    def calcZ(self, current_parameters):
        z_vector = np.array([])
        for i in range(0, self.num_data):
            eta = self.design_matrix[i]*current_parameters
            z_vector = np.append(z_vector, np.dot(self.design_matrix[i], current_parameters) + (self.target_vector[i] - self.mean_fucntion(current_parameters, self.design_matrix[i]))
            * self.calcMeanFunctionDerivative(current_parameters, self.design_matrix[i]))
        return z_vector

    def informationMatrix(self, current_parameters, W):
        designByW = np.matmul(self.design_matrix.transpose(), W)
        information = np.matmul(designByW, self.design_matrix)
        return information

    def calcRightSide(self, current_parameters, W):
        z = self.calcZ(current_parameters)
        designByW = np.matmul(self.design_matrix.transpose(), W)
        right_side = np.matmul(designByW, z)
        return right_side
        
    def solveWeightedLeastSquares(self, current_parameters):
        W = self.calcW(current_parameters)
        transform = self.informationMatrix(current_parameters, W)
        linear_target = self.calcRightSide(current_parameters, W)
        new_parameters = np.linalg.solve(transform, linear_target)
        return new_parameters

    def itterativeLeastSquares(self):
        if self.param_init == None:
            old_parameters = np.array([1 for i in range(0, self.num_param)])
        else:
            old_parameters = self.param_init

        for i in range(0, self.max_itter):
            has_not_converged = False
            new_parameters = self.solveWeightedLeastSquares(old_parameters)
            convergence_vector = abs(old_parameters - new_parameters)
            for parameter_difference in convergence_vector:
                if parameter_difference > self.convergence_limit:
                    has_not_converged = True
                    break
            old_parameters = new_parameters
            if(has_not_converged == False):
                return new_parameters
        print("Has not converged, has reached max itteration, will return most recent parameters")
        return new_parameters

    def predict(self, new_data):
        predictions = np.array([])
        for row in new_data:
            predictions = np.append(predictions, self.mean_fucntion(self.parameters, new_data[row]))
        return predictions
                

          

def main():

    # two_dim_test = pd.DataFrame({'x1': [1,2,3,4,5], 'x2': [1,1,1,1,1], 'target': [3,5,7,9,11]})
    # test_param = np.array([1,1])

    # test_linear_model = GeneralLinearModel(two_dim_test, 'linear', 'target')
    # W = test_linear_model.calcW(test_param)
    # info_mat = test_linear_model.informationMatrix(test_param, W)

    # print(test_linear_model.parameters)


    happieness_data = pd.read_csv(r'C:\Users\jrsaf\Desktop\School\Machine Learning\Projects\happiness.csv')
    happieness_data = happieness_data.drop(columns=['Overall rank', 'Country or region'])

    happiness_linear_model = GeneralLinearModel(happieness_data, 'linear', 'Score')
    print(happiness_linear_model.parameters)

    predictions = np.array([])

    for i in range(0, happiness_linear_model.design_matrix.shape[0]):
        predictions = np.append(predictions, happiness_linear_model.mean_fucntion(happiness_linear_model.parameters, happiness_linear_model.design_matrix[i]))

    rmse = np.sqrt((1/happiness_linear_model.design_matrix.shape[0])) * np.sqrt(sum((predictions - happiness_linear_model.target_vector)**2))

    print(rmse)
    
if __name__ == "__main__":
    main()