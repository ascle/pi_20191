# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import cmath
import numpy as np
import copy

from PIL import Image


PI2 = cmath.pi * 2.0


# 1 - usar sempre o dtype int8 OK
def abrir(path):
	if type(path) is str:
		return np.asarray(Image.open(path))
	else:
		raise TypeError("Path must be a string")
		

# 2 - se ligar no grayscale (color map), imagens pequenas (interpolação nearest) OK
def mostrar(img, titulo="Imagem"):
	if type(img) is np.ndarray:
		if ncanais(img) == 1:
			plt.imshow(img, interpolation='nearest', cmap=cm.gray)
		else:
			plt.imshow(img, interpolation='nearest')
		plt.title(titulo)
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
# (imgx, imgy)
def size(img):
	if type(img) is np.ndarray:
		return np.array([img.shape[1], img.shape[0]])
	else:
		raise TypeError("Image must be a numpy.ndarray")


# 5 - rgb2gray g = 0,299r + 0,587g + 0,114b + 0,0a     entrada imagem  retorna imagem
def rgb2gray(img):
	if type(img) is np.ndarray:
		if ncanais(img) == 1:
			return copy.deepcopy(img)
		else:
			img_clone = np.zeros((img.shape[0], img.shape[0]))
			for r in range(0, size(img_clone)[1]-1):
				for c in range(0, size(img_clone)[0]-1):
					img_clone[r, c] = (img[r, c][0] * 0.299) + (img[r, c][1] * 0.587) + (img[r, c][2] * 0.114)

		return img_clone
	else:
		raise TypeError("Image must be a numpy.ndarray")


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


# Discrete fourier transform 1D
def dft_1D(_array):
	if type(_array) is np.ndarray:
		retornar = []
		len_x = len(_array)

		for k in range(len_x):
			a = 0
			for n in range(len_x):
				a += _array[n]*cmath.exp(-2j*cmath.pi*k*n*(1/len_x))
			retornar.append(a)

		return np.array(retornar)
	else:
		raise TypeError("Image must be a numpy.ndarray")


# Inverse discrete fourier transform 1D
def idft_1D(_array):
	if type(_array) is np.ndarray:
		retornar = []
		len_x = len(_array)
		for n in range(len_x):
			a = 0
			for k in range(len_x):
				a += _array[k]*cmath.exp(2j*cmath.pi*k*n*(1/len_x))
			a /=len_x
			retornar.append(a)
		return np.array(retornar)
	else:
		raise TypeError("Image must be a numpy.ndarray")


# Discrete fourier transform 2D com 1 canal
def dft_2D_gray_scale(_array):
	if type(_array) is np.ndarray:
		if ncanais(_array) != 1:
			raise TypeError("Image must be 1 channel")

		global len_x, len_y
		(len_x, len_y) = size(_array)
		retornar = [[0.0 for k in range(len_x)] for l in range(len_y)]

		for k in range(len_x):
			for l in range(len_y):
				sum_red = 0
				for m in range(len_x):
					for n in range(len_y):
						e = cmath.exp(- 1j * PI2 * ((k * m) / len_x + (l * n) / len_y))
						sum_red += _array[m, n] * e
				retornar[k][l] = sum_red / len_x / len_y

		return np.array(retornar)
	else:
		raise TypeError("Image must be a numpy.ndarray")


# Inverse discrete fourier transform 2D  com 1 canal
def idft_2D_gray_scale(_array):
	if type(_array) is np.ndarray:
		global len_x, len_y
		(len_x, len_y) = size(_array)
		retornar = [[0.0 for k in range(len_x)] for l in range(len_y)]

		for m in range(len_x):
			for n in range(len_y):
				sum_red = 0.0
				for k in range(len_x):
					for l in range(len_y):
						e = cmath.exp(1j * PI2 * ((k * m) / len_x + (l * n) / len_y))
						sum_red += _array[k][l] * e
				retornar[m][n] = int(sum_red.real + 0.5)

		return np.array(retornar)
	else:
		raise TypeError("Image must be a numpy.ndarray")

   
def fft_rec_1D(_array):
	len_x = len(_array)
	if len_x <= 1:
		return _array
	else:
		even = fft_rec_1D([_array[i] for i in range(0, len_x, 2)])
		odd = fft_rec_1D([_array[i] for i in range(1, len_x, 2)])
		combined = [0] * len_x
		for m in range(len_x//2):
			t = even[m]
			#t = _array[m]
			combined[m] = t + cmath.exp(( -1j * PI2 * m) / len_x) * odd[m] #_array[int(len_x/2)] #Fodd[m]
			combined[m + len_x//2] = t - cmath.exp(( -1j * PI2 * m) / len_x) * odd[m] #_array[int(len_x/2)] #Fodd[m]

		return combined





















