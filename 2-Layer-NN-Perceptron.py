import numpy as np
#Create Feature Set and Observed Output Set
feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
labels = np.array([[1,0,0,1,1]])
labels = labels.reshape(5,1)

#Create random weights and bias
np.random.seed(42)
weights = np.random.rand(3,1)
bias = np.random.rand(1)
lr = 0.05

#Define sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))
#Define Sigmoid Derivative
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

#Train NN
for epoch in range(20000):
    inputs = feature_set
    # x.w + b
    XW = np.dot(feature_set,weights) + bias
    z = sigmoid(XW)
    
    # Calculate Error
    error = z - labels
    
    print(error.sum())
    
    #d(cost_func)/d(prediction) = error
    dcost_dpred = error
    dpred_dz = sigmoid_der(z)

    #By Chain Rule:
    #                            (dcost_dpred)              (dpred_dz)  (feature_set.T)
    #d(cost_func)/d(w) = (d(cost_func)/d(prediction)) * (d(prediction)/d(z)) * (d(z)/d(w))
        
    z_delta = dcost_dpred*dpred_dz

    #d(z)/d(w)
    inputs = feature_set.T

    #Gradient Descent
    weights -= lr*np.dot(inputs,z_delta)

    #For Bias
    for num in z_delta:
        bias -= lr*num

#Test Point
single_point = np.array([1,0,0])
result = sigmoid(np.dot(single_point,weights) + bias)
print(result)



















    


