########################################
#
# Nome: Asclepiades
# Matricula: 200920001170
# E­mail: ascle.ufs@gmail.com
#
########################################

########################################
#
# Referências
# 
# http://blog.o1iver.net/2011/12/06/python-discrete-fourier-transformation.html
# http://code.activestate.com/recipes/578997-2d-discrete-fourier-transform/
# https://en.wikipedia.org/wiki/FFT_algorithm
# https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm#Pseudocode
# https://qist.github.com/wilem/9824988
# https://gist.github.com/bellbind/1505153
# https://gist.github.com/lukicdarkoo/1ab6a9a7b24025cb428a
# https://github.com/podorozhny/fft-python/blob/master/main.py
# https://gist.github.com/lukicdarkoo/1ab6a9a7b24025cb428a
# http://ideone.com/KpcCVG
# https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
# https://pt.wikipedia.org/wiki/Transformada_de_Fourier_de_tempo_discreto
# https://rosettacode.org/wiki/Fast_Fourier_transform#Python
# https://www.youtube.com/watch?v=E1UInX_yi18
# https://www.youtube.com/watch?v=l9-DbIPZqoM
# https://www.geeksforgeeks.org/iterative-fast-fourier-transformation-polynomial-multiplication/
# https://www.analog.com/media/en/training-seminars/design-handbooks/MixedSignal_Sect5.pdf
# http://www.introni.it/pdf/Ramirez%20-%20The%20FFT%20Fundamentals%20and%20Concepts%20-%20Tektronix%201975.pdf
# 
########################################


import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image


def imread(path):
	if type(path) is str:
		return np.asarray(Image.open(path))
	else:
		raise TypeError("Path must be a string")
		
# função auxiliar		
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
		
		
def imreadgray(path):
	if type(path) is str:
		return rgb2gray(np.asarray(Image.open(path)))
	else:
		raise TypeError("Path must be a string")		
		
		
def imshow(img, titulo="Imagem"):
	if type(img) is np.ndarray:
		if ncanais(img) == 1:
			plt.imshow(img, interpolation='nearest', cmap=cm.gray)
		else:
			plt.imshow(img, interpolation='nearest')
		plt.title(titulo)
		plt.show()
	else:
		raise TypeError("Image must be a numpy.ndarray")
		
# função auxiliar		
def size(img):
	if type(img) is np.ndarray:
		return np.array([img.shape[1], img.shape[0]])
	else:
		raise TypeError("Image must be a numpy.ndarray")	

# função auxiliar
def ncanais(img):
	if type(img) is np.ndarray:
		try:
			return img.shape[2]
		except IndexError:
			return 1
	else:
		raise TypeError("Image must be a numpy.ndarray")		


# Q.01
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


# Q.01
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


# Q.04   
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
			combined[m] = t + cmath.exp(( -1j * PI2 * m) / len_x) * odd[m] 
			combined[m + len_x//2] = t - cmath.exp(( -1j * PI2 * m) / len_x) * odd[m] 

		return combined
		