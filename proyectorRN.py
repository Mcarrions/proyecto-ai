import cv2
import numpy as np
from matplotlib import pyplot as plt

class NeuralNetwork():
    
    def __init__(self):
        # seeding para numero random del 1-0 para los pesos iniciales
        np.random.seed(1)
        
        #convierte los pesos a una matriz de 3 por 1 con valores del 1 a -1 y la media de 0
        self.synaptic_weights = 2 * np.random.random((2, 1)) - 1

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

#funcion para procesar la imagen a RGB
def getImg():
    #618/1200 imagen
    img = cv2.imread('rs1.jpg',0)
    scale_percent = 5 #porciento de escala
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    #rescaled
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    edges = cv2.Canny(resized,100,100)

    #ALTURA Y GROSOR DE LA IMAGEN
    height = edges.shape[0]
    width = edges.shape[1] 

        #color del pixel individual de la imagen, comienza desde 0
        #px = edges[width,height]
        #declarando array de pixeles
        #px[119][6] = px
    w, h = width, height

    px = [[0 for f in range(w)]for g in range(h)]

    
        #asignar 
    for x in range(width-1):
        for y in range(height-1):
            if edges[y,x] == 255:
                    px[y][x] = 1
            else:
                px[y][x] = edges[y,x]
    
    #abrir el archivo
    np.savetxt('output.txt', px, fmt='%.2e')
        
    return px


if __name__ == "__main__":
    #inicializacion de la clase neuron
    neural_network = NeuralNetwork()
    print("Calculando pesos aleatorios: ")
    print(neural_network.synaptic_weights)

    #data de entrenamiento de entradas y 1 salida con la respuesta esperada
    training_inputs = np.array([[0,0],
                                [1,1],
                                [1,0],
                                [0,1]])

    training_outputs = np.array([[0,1,1,0]]).T
    #training_inputs = np.array(getImg())
    training_outputs = np.array([[0,1]]).T
    iterations = int(input("Iteraciones:"))
    #loop de entrenamiento entradas, salidas, iteraciones de aprendizaje
    neural_network.train(training_inputs, training_outputs, iterations)

    print("Pesos de capa oculta entrenados: ")
    print(neural_network.synaptic_weights)	
    user_input_one = 0
    user_input_two = 1
   
    #re=neural_network.think(np.array(getImg()))
    re=neural_network.think(np.array([user_input_one, user_input_two, user_input_three]))
    print("Resultado: ")
    
    #if neural_network.think(np.array(getImg()))>0.5:
    if neural_network.think(np.array([user_input_one, user_input_two]))>0.5:
        print("Es una lata")
    else:
        print("No es una lata")
    matrix = getImg()
    print(np.matrix(matrix))
    