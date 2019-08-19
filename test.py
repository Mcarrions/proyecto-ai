import cv2
import numpy as np
from matplotlib import pyplot as plt


#funcion para procesar la imagen a RGB
def getImg():
    print("test")
    #618/1200 imagen
    img = cv2.imread('rs1.jpg',0)
    scale_percent = 1 # percent to scale
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
            px[y][x] = edges[y,x]
            
    return px
	

if __name__ == "__main__":
    matrix = getImg()
    print(matrix)