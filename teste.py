# -*- coding: utf-8 -*-

import pi_ascle as cl
import numpy as np

from numpy.fft import fft


IMG_LENNA_JPG = 'imagens/lena.jpeg'
IMG_LENNA_50  = 'imagens/lena50.jpeg'	# width(largura, X ):46 heigth(comprimento, Y):41
IMG_LENNA_50_G  = 'imagens/lenna50G.jpg'
IMG_LENNA_PNG = 'imagens/lena.png'
IMG_LENNA_GS  = 'imagens/lenna_grey.jpg'


# teste função 5
def rgb2gray_test():
	lena_jpg = cl.abrir(IMG_LENNA_50)
	cl.mostrar(cl.rgb2gray(lena_jpg))
	cl.mostrar(lena_jpg)
	
	
# Testar DFT 1D
def dft_1D_test():
	_array = np.array([25,36,89,214,36,55,41,2,3,255], dtype=np.uint8)
	dft = cl.dft_1D(_array)
	print(_array)
	print(dft)
	print(cl.idft_1D(dft))


# Testa o dft 2D com 1 canal
def dft_2D_gray_scale_test():
	_array = np.array([[0, 30, 65],
					 [80, 118, 140],
					 [163, 200, 255]], dtype=np.uint8)

	print("\n###### array")
	print(_array)

	dft = cl.dft_2D_gray_scale(_array)
	print("\n###### OP1")
	print(dft)

	#print("\n###### OP2")
	#print(cl.FT_2D(_array))

	print("\n###### INV 1")
	print(cl.idft_2D_gray_scale(dft))	
	

def dft_2D_gray_scale_image_test():
	lena_jpg = cl.abrir(IMG_LENNA_GS)
	cl.mostrar(lena_jpg, IMG_LENNA_GS)

	lena_dft = fft(lena_jpg)
	
	lena_dft_real = np.zeros(lena_dft.shape)
	lena_dft_imag = np.zeros(lena_dft.shape)
	for r in range(0, cl.size(lena_dft)[1]-1):
		for c in range(0, cl.size(lena_dft)[0]-1):
			lena_dft_real[r, c] = lena_dft[r, c].real
			lena_dft_imag[r, c] = lena_dft[r, c].imag
	
	cl.mostrar(lena_dft_real, "REAL - "+IMG_LENNA_GS)
	cl.mostrar(lena_dft_imag, "IMAG - "+IMG_LENNA_GS)
	
	
# Testar FFT 1D
def fft_rec_1D_test():
	_array = np.array([25,36,89,214,36,55,41,2,3,255], dtype=np.uint8)
	dft = cl.dft_1D(_array)
	_fft = cl.fft_rec_1D(_array)
	
	print("\n###### ARRAY")
	print(_array)
	
	print("\n###### DFT")
	print(dft)
	
	print("\n###### FFT")
	print(np.array(_fft))
	
	print("\n###### FFT NUMPY")
	print(fft(_array))
	
	
dft_2D_gray_scale_image_test()



























