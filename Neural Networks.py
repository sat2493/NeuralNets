class Network:
    def __init__(self, x, batch_num, numOutputNeurons):
        # Number of Neurons in Each Layer
        self.numOutputNeurons = numOutputNeurons
        
        # Initialize Input & Output Layers
        
        self.inputlayer = self.InputLayer(x[batch_num])
        
        # Initialize succeeding layers, up until the y layer is reached
        # Add Hidden Layers in between
        self.outputlayer = self.NextLayer(self.numOutputNeurons, x[batch_num].size)
        
    def calculate_activation(self, prevLayer, currLayer):
        # Matrix-vector multiplication to calculate next layer values
        currLayer.neurons = np.matmul(currLayer.weights, prevLayer.neurons)
        # Subtract Current Layer's Biases from Neurons
        currLayer.neurons = currLayer.neurons - currLayer.biases
        # Apply the sigmoid algorithm to the result
        
        
    def decision(self):
        neuron_activation = self.outputlayer.neurons[0]
        max_number = 0
        number = 0
 
        for i in self.outputlayer.neurons:
            if i > neuron_activation:
                max_number = number
            number = number + 1
            
        return max_number    
                
    
    class InputLayer:
        # Read image pixels, initialize neurons with the values of each image
        
        def __init__(self, x):
            # self.neurons = np.ones(numNeurons)
            # Initialize empty array
            self.neurons = []
            for i in x:
                for pixel in i:
                    self.neurons.append(pixel)
            self.neurons = np.array(self.neurons)
            
    class NextLayer: 
        def __init__(self, numCurrLayerNeurons, numPrevLayerNeurons):
            # Initialize layer's neurons and biases
            self.neurons = np.ones(numCurrLayerNeurons)
            self.biases = np.ones(numCurrLayerNeurons)
            
            # Initialize weights
            self.weights = []
            for i in range(numCurrLayerNeurons):
                # np.append(self.weights, np.empty(numPrevLayerNeurons))  
                self.weights.append(np.zeros(numPrevLayerNeurons))
            #np.asarray(self.weights)
            self.weights = np.asarray(self.weights)
