# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import cmath
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
			for r in range(0, size(img_clone)[1]-1):
				for c in range(0, size(img_clone)[0]-1):
					#temp = copy.deepcopy(img_clone[r, c])
					img_clone[r, c] = np.array([np.uint8((img_clone[r, c][0] * 0.299) + (img_clone[r, c][1] * 0.587) + (img_clone[r, c][2] * 0.114))], dtype=np.uint8)

		return img_clone
	else:
		raise TypeError("Image must be a numpy.ndarray")


# teste função 5
def rgb2gray_test():
	lena_jpg = abrir(IMG_LENNA_50)
	mostrar(rgb2gray(lena_jpg))
	mostrar(lena_jpg)


# Questão 18 - Crie uma função chamada seSquare3, que retorna o elemento estruturante binário [[1,1, 1], [1, 1, 1], [1, 1, 1]].
def seSquare3():
	#return np.full((3,3), 1)
	return np.array([[1,1, 1],
					 [1, 1, 1],
					 [1, 1, 1]], dtype=np.uint8)


# Questão 19 - Crie uma função chamada seCross3, que retorna o elemento estruturante binário [[0, 1,0], [1, 1, 1], [0, 1, 0]].
def seCross3():
	return np.array([[0, 1,0],
					 [1, 1, 1],
					 [0, 1, 0]], dtype=np.uint8)


# Discrete fourier transform
def dft_1D(x):
    t = []
    N = len(x)
    for k in range(N):
        a = 0
        for n in range(N):
            a += x[n]*cmath.exp(-2j*cmath.pi*k*n*(1/N))
        t.append(a)
    return t

# Inverse discrete fourier transform
def idft_1D(t):
    x = []
    N = len(t)
    for n in range(N):
        a = 0
        for k in range(N):
            a += t[k]*cmath.exp(2j*cmath.pi*k*n*(1/N))
        a /= N
        x.append(a)
    return x

def dft_1D_test():
	array = [25,36,89,214,36,55,41,2,3,255]
	dft = dft_1D(array)
	print(array)
	print(dft)
	print(idft_1D(dft))


dft_1D_test()