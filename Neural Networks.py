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
        # Matrix-vector multiplication to calculate next layer values
        currLayer.neurons = np.matmul(currLayer.weights, prevLayer.neurons)
        # Subtract Current Layer's Biases from Neurons
        currLayer.neurons = currLayer.neurons 
        # Apply the sigmoid function to the result
        currLayer.neurons = 1 / ( np.ones(currLayer.neurons.size) + pow(np.e, -currLayer.neurons) )
        
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
            self.neurons = np.ones(numNeurons)
            
        def read(self, x):
            count = 0
            for i in x:
                for pixel in i:
                    self.neurons[count] = pixel
                    count = count + 1
            
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
