import pandas as pd
import numpy as np
import tensorflow as tf

dataset = tf.keras.datasets.mnist.load_data(path="mnist.npz")

x_train = dataset[0][0]
y_train = dataset[0][1]

x_test = dataset[1][0]
y_test = dataset[1][1]

class Network:
    def __init__(self, numNeuronsInEachLayer):
        
        # Initialize Input Layer
        self.inputlayer = self.InputLayer(numNeuronsInEachLayer[0])
        self.layers = [self.inputlayer]
        
        # Initialize each succeeding layer based on the one before it
        
        # Have a counter to keep track of which layer we are at
        count = 0
        for numNeurons in numNeuronsInEachLayer:
            # Skip the input layer, which has no preceding layer
            if count == 0:
                count = count + 1
                continue
            nextlayer = self.NextLayer(numNeurons, numNeuronsInEachLayer[count - 1])
            self.layers.append(nextlayer)
            count = count + 1
            
        self.outputlayer = self.layers[len(self.layers) - 1]
            
    def update(self, x):
        
        count = 0
        for layer in self.layers:
            # Read inputs onto input layer
            if layer == self.inputlayer:
                layer.read(x)
            # Update other layers based on the neurons from the previous layer
            else:
                layer = self.calculate_activations(self.layers[count - 1], layer)
            count = count + 1
        
    def calculate_activations(self, prevLayer, currLayer):
        # Calculate z value
        currLayer.z_values = z_value(prevLayer, currLayer) 
        # Apply the sigmoid function to the result
        currLayer.neurons = sigmoid(currLayer.z_values)
            
    def decision(self):
        lastlevel = len(self.layers) - 1
        outputlayer = self.layers[lastlevel]
        neuron_activation = outputlayer.neurons[0]
        max_number = 0
        number = 0
 
        for i in outputlayer.neurons:
            if i > neuron_activation:
                max_number = number
            number = number + 1
            
        return max_number    
                
    
    class InputLayer:
        # Read image pixels, initialize neurons with the values of each image
        
        def __init__(self, numNeurons):
            self.z_values = np.ones(numNeurons)
            self.neurons = np.ones(numNeurons)
            
        def read(self, x):
            count = 0
            for i in x:
                for pixel in i:
                    self.z_values[count] = pixel
                    self.neurons = sigmoid(self.z_values)
                    count = count + 1
            
    class NextLayer: 
        def __init__(self, numCurrLayerNeurons, numPrevLayerNeurons):
            # Initialize layer's neurons and biases
            self.z_values = np.ones(numCurrLayerNeurons)
            self.neurons = np.ones(numCurrLayerNeurons)
            self.biases = np.ones(numCurrLayerNeurons)
            
            # Initialize weights
            self.weights = []
            for i in range(numCurrLayerNeurons):
                # np.append(self.weights, np.empty(numPrevLayerNeurons))  
                self.weights.append(np.zeros(numPrevLayerNeurons))
            #np.asarray(self.weights)
            self.weights = np.asarray(self.weights)

class Table:
    def __init__(self, network, batchsize):
        self.tables = []
        
        count = 0
        for layer in network.layers:
            if layer == network.inputlayer:
                count = count + 1
                continue
                
            self.tables.append(self.LayerTable(network.layers[count-1], layer, batchsize))
            count = count + 1
            
    class LayerTable:
        # Initialize Tables & Store Them Inside the Network
        def __init__(self, prevLayer, currLayer, batchsize):
            # Set table width to be equal to batch size
            self.df_bias = pd.DataFrame(columns=range(batchsize))
            self.df_weight = pd.DataFrame(columns=range(batchsize))
            self.df_prev_neuron = pd.DataFrame(columns=range(batchsize))

            # Set table height to correspond to respective curr/prev neuron positions
            self.df_weight[0] = np.empty(len(currLayer.neurons), dtype=object) #np.ones(len(currLayer.neurons)) 

def z_value(prevLayer, currLayer):
    # Matrix-vector multiplication to calculate next layer values
    result = np.matmul(currLayer.weights, prevLayer.neurons)
    # Subtract Current Layer's Biases from Neurons
    result = result - currLayer.biases 
    return result

def sigmoid(z_value):
    return 1 / ( np.ones(z_value.size) + pow(np.e, -z_value) )

def cost(actual, ideal):
    sum = 0
    for y, y_bar in zip(actual, ideal):
        sum = sum + pow(y - y_bar, 2)
    return sum

def derivative_of_current_neuron_with_respect_to_current_z_value(z_value):
    return ((pow(np.e, -z_value)) / pow((1 + pow(np.e, -z_value)), 2)) 

def derivative_of_cost_with_respect_to_current_neuron(a, y):
    return (2 * (a - y))

def train(network, X, Y):
    # Initialize Table
    table = Table(network, len(X))
        
    # Loop through all training examples
    training_example = 0
    for x,y in zip(X,Y):
        print("i:", training_example)
        # Run Training Example Through Network
        network.update(x)
        
        # Perform Backpropagation Across All Training Examples
        full_backpropagation(network, y, table, training_example)
        training_example = training_example + 1

    # Take averages across all training examples
    count = 1
    for t in table.tables:
        # Tweak bias parameters
        nudge_biases = np.array(t.df_bias.mean(1))
        network.layers[count].biases = network.layers[count].biases + nudge_biases

        # Tweak weight parameters
        count1 = 0
        for w in t.df_weight.values:
            nudge_weights = w.mean()
            network.layers[count].weights[count1] = network.layers[count].weights[count1] + nudge_weights
            count1 = count1 + 1
        count = count + 1

    # Add nudges to each relevant parameter

# Full Algorithm For One Training Example
def full_backpropagation(network, y, table, training_example):
    
    _y = np.zeros(len(network.outputlayer.neurons))
    _y[y] = 1

    index = len(network.layers) - 1
    while index != 0:
        currLayer = network.layers[index]
        prevLayer = network.layers[index - 1]
        lastLayer = network.layers[len(network.layers) - 1]
                    
        if index == len(network.layers) - 1:
            partial_backpropagation(currLayer, prevLayer, lastLayer, _y, table.tables[index-1], None, training_example)
        else:
            partial_backpropagation(currLayer, prevLayer, lastLayer, _y, table.tables[index-1], table.tables[index], training_example)
        # Go back one level in the neural network
        index = index - 1

# Partial Algorithm Involving Current Layer and Previous Layer
def partial_backpropagation(currLayer, prevLayer, lastLayer, ideal_outputs, layertableCurrent, layertableBefore, training_example):
    
    # Set up tables
    layertableCurrent.df_bias[training_example] = np.empty(len(currLayer.neurons))
    layertableCurrent.df_prev_neuron[training_example] = np.empty(len(prevLayer.neurons))
    for i in range(len(currLayer.neurons)):
        layertableCurrent.df_weight[training_example].loc[i] = np.array([])

    # Begin backpropagation
    if currLayer == lastLayer:
        j = 0
        for a_j, z_j, y in zip(currLayer.neurons, currLayer.z_values, ideal_outputs):
            dC_da = derivative_of_cost_with_respect_to_current_neuron(a_j, y)
            da_dz = derivative_of_current_neuron_with_respect_to_current_z_value(z_j)
            dz_db = 1

            nudge_bias = dC_da * da_dz * dz_db
            layertableCurrent.df_bias[training_example].loc[j] = nudge_bias

            k = 0
            for a_k, w_jk in zip(prevLayer.neurons, currLayer.weights[j]):
                dz_dw = a_k
                dz_da_minus1 = w_jk

                nudge_weight = dC_da * da_dz * dz_dw
                #print(nudge_weight, layertableCurrent.df_weight[training_example].loc[j])
                layertableCurrent.df_weight.loc[:,(training_example,j)] = np.append(layertableCurrent.df_weight[training_example].loc[j], nudge_weight)
                nudge_prev_neuron = dC_da * da_dz * dz_da_minus1
                layertableCurrent.df_prev_neuron[training_example][k] = layertableCurrent.df_prev_neuron[training_example][k] + nudge_prev_neuron

                k = k + 1
            j = j + 1

    else:
        j = 0
        for a_j, z_j, dC_da in zip(currLayer.neurons, currLayer.z_values, layertableBefore.df_prev_neuron[training_example]):
            da_dz = derivative_of_current_neuron_with_respect_to_current_z_value(z_j)
            dz_db = 1

            nudge_bias = dC_da * da_dz * dz_db
            layertableCurrent.df_bias[training_example].loc[j] = nudge_bias

            k = 0
            for a_k, w_jk in zip(prevLayer.neurons, currLayer.weights[j]):
                dz_dw = a_k
                dz_da_minus1 = w_jk

                nudge_weight = dC_da * da_dz * dz_dw
                layertableCurrent.df_weight.loc[:,(training_example,j)] = np.append(layertableCurrent.df_weight[training_example].loc[j], nudge_weight)
        

                nudge_prev_neuron = dC_da * da_dz * dz_da_minus1
                layertableCurrent.df_prev_neuron[training_example][k] = layertableCurrent.df_prev_neuron[training_example][k] + nudge_prev_neuron 
                
                k = k + 1
            j = j + 1

def model_parameters(numInputNeurons, numMiddleNeurons, numOutputNeurons):
    numNeuronsInEachLayer = [numInputNeurons]
    for i in numMiddleNeurons:
        numNeuronsInEachLayer.append(i)
    numNeuronsInEachLayer.append(numOutputNeurons)

    return numNeuronsInEachLayer

# Set up parameters, and build the network
model_parameters = model_parameters(784,[16,16,16],10)
n = Network(model_parameters)

# Define batch size parameters
i = 0
j = 1000

# Train the model
while j <= 60000:
    print(i, ":", j)
    X = x_train[i:j]
    Y = y_train[i:j]

    train(n,X,Y)

    i = i + 1000
    j = j + 1000
