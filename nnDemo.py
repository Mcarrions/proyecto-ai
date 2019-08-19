import cv2
import numpy as np
from matplotlib import pyplot as plt

class NeuralNetwork():
    
    def __init__(self):
        # seeding para numero random
        np.random.seed(1)
        
        #convierte los pesos a una matriz de 3 por 1 con valores del 1 a -1 y la media de 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        #formula sigmoide
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        #derivada sigmoide
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        
        #entrena al modelo y pesos
        for iteration in range(training_iterations):
            #envia los datos de la neurona
            output = self.think(training_inputs)

            #error del backpropagation
            error = training_outputs - output
            
            #ajuste de pesos
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):
        #pasa las entradas de las neuronas iniciales  
        #convierte los valores a floats
        
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

class NewName():
	def __init__(self):
		print("test")
	

if __name__ == "__main__":
    #inicializacion de la clase neuron
    neural_network = NeuralNetwork()
	
    print("Pesos al azar: ")
    print(neural_network.synaptic_weights)

    #data de entrenamiento de 3 entradas y 1 salida
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T

    #loop de entrenamiento entradas, salidas, iteraciones de aprendizaje
    neural_network.train(training_inputs, training_outputs, 10000)

    print(neural_network.synaptic_weights)	
    user_input_one = str(input("1: "))
    user_input_two = str(input("2: "))
    user_input_three = str(input("3: "))
    re=neural_network.think(np.array([user_input_one, user_input_two, user_input_three]))
    print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three])))
	
	
	    
	

	