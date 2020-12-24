class Network:
    def __init__(self, numNeuronsInEachLayers):
        
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

# Finds derivative with respect to a single weight parameter
def derivative_of_cost_with_respect_to_weight(current_neuron, previous_neuron, z_value, ideal_output):
    return previous_neuron * ((pow(np.e, -z_value)) / pow((1 + pow(np.e, -z_value)), 2)) * (2 * (current_neuron - ideal_output))

# Finds derivative with respect to a single bias parameter
def derivative_of_cost_with_respect_to_bias(current_neuron, z_value, ideal_output):
    return ((pow(np.e, -z_value)) / pow((1 + pow(np.e, -z_value)), 2)) * (2 * (current_neuron - ideal_output))

# Finds derivative with respect to a single weight parameter
def derivative_of_cost_with_respect_to_previous_neuron(weight, current_neuron, z_value, ideal_output):
    return weight * ((pow(np.e, -z_value)) / pow((1 + pow(np.e, -z_value)), 2)) * (2 * (current_neuron - ideal_output))

def partial_backpropagation(currLayer, prevLayer, ideal_outputs, df_bias, df_weight, df_prev_nueron, train_number):
    
    j = 0
    for a_j, z_j, y, w_j in zip(currLayer.neurons, currLayer.z_values, ideal_outputs, currLayer.weights):
        nudge_bias = derivative_of_cost_with_respect_to_bias(a_j, z_j, y)
        # Store Nudge in Bias Table
        df_bias[train_number][j] = nudge_bias
        
        for a_k, w_jk in zip(prevLayer.neurons, w_j):
            nudge_weight = derivative_of_cost_with_respect_to_weight(a_j, a_k, z_j, y)
            # Store Nudge in Weight Table
            np.append(df_weight[train_number][j], nudge_weight)
            
            nudge_a_k = derivative_of_cost_with_respect_to_previous_neuron(a_j, z_j, y)
            # Record Nudge in Previous Weight Table
            df_prev_nueron[train_number][j] = nudge_a_k
            
        # Move on to next neuron in L layer
        j = j + 1   
