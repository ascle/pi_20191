# -*- coding: utf-8 -*-

import pi_ascle as cl
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


def questao3(_nome_imagem = "lena1.jpg"):
	imagem = cl.rgb2gray(cl.abrir("exercicio_fft/" + _nome_imagem))
	lena_dft = fft(imagem)
	lena_ifft = ifft(lena_dft)

	# pega a parte real e a partye imaginaria
	lena_dft_real = np.zeros(lena_dft.shape)
	lena_dft_imag = np.zeros(lena_dft.shape)
	for r in range(0, cl.size(lena_dft)[1] - 1):
		for c in range(0, cl.size(lena_dft)[0] - 1):
			lena_dft_real[r, c] = lena_dft[r, c].real
			lena_dft_imag[r, c] = lena_dft[r, c].imag

	# convere a ifft em imagem
	lena_ifft_real = lena_dft_real = np.zeros(lena_ifft.shape)
	for r in range(0, cl.size(lena_ifft)[1] - 1):
		for c in range(0, cl.size(lena_ifft)[0] - 1):
			# print(type(lena_ifft[r, c]))
			# print(lena_ifft[r, c])
			lena_ifft_real[r, c] = int(lena_ifft[r, c].real + 0.5)

	cl.mostrar(imagem, _nome_imagem)
	cl.mostrar(lena_dft_real, "REAL - " + _nome_imagem)
	cl.mostrar(lena_dft_imag, "IMAG - " + _nome_imagem)
	cl.mostrar(lena_ifft_real, "INVERSA - " + _nome_imagem)


def questao_2(_nome_imagem):
	imagem = cl.abrir("exercicio_fft/"+_nome_imagem)
	lena_dft = fft(imagem)
	lena_ifft = ifft(lena_dft)
	
	# pega a parte real e a partye imaginaria
	lena_dft_real = np.zeros(lena_dft.shape)
	lena_dft_imag = np.zeros(lena_dft.shape)
	for r in range(0, cl.size(lena_dft)[1]-1):
		for c in range(0, cl.size(lena_dft)[0]-1):
			lena_dft_real[r, c] = lena_dft[r, c].real
			lena_dft_imag[r, c] = lena_dft[r, c].imag
			
	#convere a ifft em imagem
	lena_ifft_real = lena_dft_real = np.zeros(lena_ifft.shape)
	for r in range(0, cl.size(lena_ifft)[1]-1):
		for c in range(0, cl.size(lena_ifft)[0]-1):
			print(type(lena_ifft[r, c]))
			print(lena_ifft[r, c])
			lena_ifft_real[r, c] = int(lena_ifft[r, c].real + 0.5)
	
	cl.mostrar(imagem, _nome_imagem)
	cl.mostrar(lena_dft_real, "REAL - "+_nome_imagem)
	cl.mostrar(lena_dft_imag, "IMAG - "+_nome_imagem)
	cl.mostrar(lena_ifft_real, "INVERSA - "+_nome_imagem)


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
	
	
questao3()









	
