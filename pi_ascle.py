# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

import numpy as np

from PIL import Image

import copy

IMG_LENNA_JPG = 'imagens/lena.jpeg'
# width(largura, X ):46 heigth(comprimento, Y):41
IMG_LENNA_50 =  'imagens/lena50.jpeg'
IMG_LENNA_PNG = 'imagens/lena.png'
IMG_LENNA_GS =  'imagens/lenna_grey.jpg'



# 1 - usar sempre o dtype int8 OK
def abrir(path):
	if type(path) is str:
		return np.asarray(Image.open(path))
	else:
		raise TypeError("Path must be a string")
		
		

# 2 - se ligar no grayscale (color map), imagens pequenas (interpolação nearest) OK
def mostrar(img):
	if type(img) is np.ndarray:
		if ncanais(img) == 1:
			plt.imshow(img, interpolation='nearest', cmap=cm.gray)
		else:
			plt.imshow(img, interpolation='nearest')
		plt.show()
	else:
		raise TypeError("Image must be a numpy.ndarray")
		
		

# 3 - se for uma imagem grayscale OK
def ncanais(img):
	if type(img) is np.ndarray:
		try:
			return img.shape[2]
		except IndexError:
			return 1
	else:
		raise TypeError("Image must be a numpy.ndarray")


# 4 - Crie uma função chamada size que retorna um vetor onde a primeira posição é a largura
# e a segunda é a altura em pixels da imagem de entrada.
def size(img):
	if type(img) is np.ndarray:
		return np.array([img.shape[1], img.shape[0]])
	else:
		raise TypeError("Image must be a numpy.ndarray")



# 5 - rgb2gray g = 0,299r + 0,587g + 0,114b + 0,0a     entrada imagem  retorna imagem
def rgb2gray(img):
	img_clone = copy.deepcopy(img)
	if type(img_clone) is np.ndarray:
		if ncanais(img_clone) == 1:
			return img_clone
		else:
			for r in range(0, size(img_clone)[0]-1):
				for c in range(0, size(img_clone)[1]-1):
					temp = copy.deepcopy(c)
					print(c)
					img_clone[r, c] = np.array([(temp[0] * 0.299) + (temp[1] * 0.587) + (temp[2] * 0.114)], dtype=np.uint8)
					print(img_clone[r, c])

		return img_clone
	else:
		raise TypeError("Image must be a numpy.ndarray")




# 6 - imreadgray entrada nome do arquivo  retorna imagem

#if __name__ == "__main__":
#lena_jpg = abrir(IMG_LENNA_JPG)
#lena_png = abrir(IMG_LENNA_PNG)
#mostrar(lena_jpg)
#mostrar(lena_png)


#lena_gs = abrir(IMG_LENNA_50)
#mostrar(lena_gs)
#print(type(lena_jpg.shape))
#print(lena_jpg.shape)
#print(type(size(lena_jpg)))
#print(lena_jpg)

lena_jpg = abrir(IMG_LENNA_50)
mostrar(lena_jpg)
mostrar(rgb2gray(lena_jpg))

