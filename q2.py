# -*- coding: utf-8 -*-

import asclepiades_neto as cl
import numpy as np

from numpy.fft import fft
from numpy.fft import ifft
from datetime import datetime


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

# Diagrama de módulo
# Diagrama de fase
# Qual é o dominio da frequencia de uma imagem?
def questao_3(_nome_imagem = "lena1.jpg"):
	imagem = cl.rgb2gray(cl.imread("exercicio_fft/" + _nome_imagem))
	lena_dft = fft(imagem)
	lena_ifft = ifft(lena_dft)

	global lena_dft_real, lena_dft_imag
	(lena_dft_real, lena_dft_imag) = img_real_imag(lena_dft)

	# convere a ifft em imagem
	lena_ifft_real = np.zeros(imagem.shape)
	for r in range(0, cl.size(lena_ifft)[1] - 1):
		for c in range(0, cl.size(lena_ifft)[0] - 1):
			lena_ifft_real[r, c] = int(lena_ifft[r, c].real + 0.5)

	cl.imshow(imagem, _nome_imagem)
	cl.imshow(lena_dft_real, "REAL - " + _nome_imagem)
	cl.imshow(lena_dft_imag, "IMAG - " + _nome_imagem)
	cl.imshow(lena_ifft_real, "INVERSA - " + _nome_imagem)


def questao_2(_nome_imagem):
	imagem = cl.imreadgray("exercicio_fft/"+_nome_imagem)
	lena_dft = fft(imagem)
	lena_ifft = ifft(lena_dft)

	global im_dft_real, im_dft_imag
	(im_dft_real, im_dft_imag) = img_real_imag(lena_dft)
			
	#convere a ifft em imagem
	lena_ifft_real = np.zeros(imagem.shape)
	for r in range(0, cl.size(lena_ifft)[1]-1):
		for c in range(0, cl.size(lena_ifft)[0]-1):
			lena_ifft_real[r, c] = int(lena_ifft[r, c].real + 0.5)
	
	cl.imshow(imagem, _nome_imagem)
	cl.imshow(im_dft_real, "REAL - "+_nome_imagem)
	cl.imshow(im_dft_imag, "IMAG - "+_nome_imagem)
	cl.imshow(lena_ifft_real, "INVERSA - "+_nome_imagem)


def img_real_imag(imagem):
	# pega a parte real e a partye imaginaria
	dft_real = np.zeros(imagem.shape) #, dtype=np.uint8)
	dft_imag = np.zeros(imagem.shape )
	for r in range(0, cl.size(imagem)[1]-1):
		for c in range(0, cl.size(imagem)[0]-1):
			dft_real[r, c] = imagem[r, c].real
			dft_imag[r, c] = imagem[r, c].imag

	return (dft_real, dft_imag)

def letra_a():
	questao_2(SIN2_GIF)
	
def letra_b():
	questao_2(SIN4_GIF)

def letra_c():
	questao_2(SIN4H_GIF)
	
def letra_d():
	questao_2(SIN8D_GIF)
	
def letra_e():
	questao_2(SIN10_GIF)
	 
def letra_f():
	questao_2(SIN26_GIF)
	
def letra_g():
	questao_2(SIN_COMBO_GIF)
	 
def letra_h():
	questao_2(SIN_COMBO_2_GIF)
	  
def letra_i():
	questao_2(SINX3_GIF)
	
def letra_j():
	questao_2(SIN_ALL_GIF)
  
def letra_k():
	questao_2(GAUSIAN_GIF)
	
def letra_l():
	questao_2(SIN10_GAUSS_GIF)

def print_q2():
	letra_a()
	letra_b()
	letra_c()
	letra_d()
	letra_e()
	letra_f()
	letra_g()
	letra_h()
	letra_i()
	letra_j()
	letra_k()
	letra_l()
	
	
print_q2()

#questao_3()









	
