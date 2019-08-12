# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import cmath
import numpy as np

from PIL import Image

import copy

IMG_LENNA_JPG = 'imagens/lena.jpeg'
IMG_LENNA_50  = 'imagens/lena50.jpeg'	# width(largura, X ):46 heigth(comprimento, Y):41
IMG_LENNA_PNG = 'imagens/lena.png'
IMG_LENNA_GS  = 'imagens/lenna_grey.jpg'

PI2 = cmath.pi * 2.0

GAUSIAN_GIF = 'gaussian.gif'
LENA_1_JPG = 'lena1.jpg'
SIN2_GIF = 'sin2.gif'
SIN4_GIF = 'sin4.gif'
SIN4H_GIF = 'sin4h.gif'
SIN8D_GIF = 'sin8d.gif'
SIN10_GIF = 'sin10+4h.gif'
SIN10_GAUSS_GIF = 'sin10+4h+gaus.gif'
SIN26_GIF = 'sin26.gif'
SIN_ALL_GIF = 'sin_all.gif'
SIN_COMBO_GIF = 'sincombo.gif'
SIN_COMBO_2_GIF = 'sincombo2.gif'
SINX3_GIF = 'sinx3.gif'



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

# Testar DFT 1D
def dft_1D_test():
	_array = np.array([25,36,89,214,36,55,41,2,3,255], dtype=np.uint8)
	dft = dft_1D(_array)
	print(_array)
	print(dft)
	print(idft_1D(dft))


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

		return np.frombuffer(retornar)
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

		return np.frombuffer(retornar)
	else:
		raise TypeError("Image must be a numpy.ndarray")

# Testa o dft 2D com 1 canal
def dft_2D_gray_scale_test():
	_array = np.array([[0, 30, 65],
					 [80, 118, 140],
					 [163, 200, 255]], dtype=np.uint8)

	print("\n###### array")
	print(_array)

	dft = dft_2D_gray_scale(_array)
	print("\n###### OP1")
	print(dft)

	print("\n###### OP2")
	print(FT_2D(_array))

	print("\n###### INV 1")
	print(idft_2D_gray_scale(dft))


def dft_2D_gray_scale_image_test():
	lena_jpg = abrir(IMG_LENNA_GS)
	mostrar(rgb2gray(lena_jpg))

	lena_dft = dft_2D_gray_scale(lena_jpg)
	#mostrar(lena_dft)

	lena_idft = idft_2D_gray_scale(lena_dft)
	mostrar(lena_idft)

#dft_2D_gray_scale_image_test()