# -*- coding: utf-8 -*-

import pi_ascle as cl
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


def letra_a():
    print(datetime.today())
    imagem = cl.abrir("exercicio_fft/"+SIN2_GIF)
    img_dft = cl.dft_2D_gray_scale(imagem)
    print(datetime.today())
    img_idft = idft_2D_gray_scale(img_dft)
    print(datetime.today())

    cl.mostrar(imagem, SIN2_GIF)
    cl.mostrar(img_dft, "DFT - " + SIN4_GIF)
    cl.mostrar(img_idft, "IDFT - " + SIN4_GIF)

def letra_b():
    imagem = cl.abrir("exercicio_fft/"+SIN4_GIF)
    img_dft = cl.dft_2D_gray_scale(imagem)
    img_idft = idft_2D_gray_scale(img_dft)

    cl.mostrar(imagem, SIN4_GIF)
    cl.mostrar(img_dft, "DFT - "+SIN4_GIF)
    cl.mostrar(img_idft, "IDFT - " + SIN4_GIF)

def letra_c():
    imagem = cl.abrir("exercicio_fft/"+SIN4H_GIF)
    img_dft = cl.dft_2D_gray_scale(imagem)
    img_idft = idft_2D_gray_scale(img_dft)

    cl.mostrar(imagem, SIN4H_GIF)
    cl.mostrar(img_dft, "DFT - " + SIN4_GIF)
    cl.mostrar(img_idft, "IDFT - " + SIN4_GIF)

def letra_d():
    imagem = cl.abrir("exercicio_fft/"+SIN8D_GIF)
    img_dft = cl.dft_2D_gray_scale(imagem)
    img_idft = idft_2D_gray_scale(img_dft)

    cl.mostrar(imagem, SIN8D_GIF)
    cl.mostrar(img_dft, "DFT - " + SIN4_GIF)
    cl.mostrar(img_idft, "IDFT - " + SIN4_GIF)

def letra_e():
    imagem = cl.abrir("exercicio_fft/"+SIN10_GIF)
    img_dft = cl.dft_2D_gray_scale(imagem)
    img_idft = idft_2D_gray_scale(img_dft)

    cl.mostrar(imagem, SIN10_GIF)
    cl.mostrar(img_dft, "DFT - " + SIN4_GIF)
    cl.mostrar(img_idft, "IDFT - " + SIN4_GIF)

def letra_f():
    imagem = cl.abrir("exercicio_fft/"+SIN26_GIF)
    img_dft = cl.dft_2D_gray_scale(imagem)
    img_idft = idft_2D_gray_scale(img_dft)

    cl.mostrar(imagem, SIN26_GIF)
    cl.mostrar(img_dft, "DFT - " + SIN4_GIF)
    cl.mostrar(img_idft, "IDFT - " + SIN4_GIF)

def letra_g():
    imagem = cl.abrir("exercicio_fft/"+SIN_COMBO_GIF)
    img_dft = cl.dft_2D_gray_scale(imagem)
    img_idft = idft_2D_gray_scale(img_dft)

    cl.mostrar(imagem, SIN_COMBO_GIF)
    cl.mostrar(img_dft, "DFT - " + SIN4_GIF)
    cl.mostrar(img_idft, "IDFT - " + SIN4_GIF)

def letra_h():
    imagem = cl.abrir("exercicio_fft/"+SIN_COMBO_2_GIF)
    img_dft = cl.dft_2D_gray_scale(imagem)
    img_idft = idft_2D_gray_scale(img_dft)

    cl.mostrar(imagem, SIN_COMBO_2_GIF)
    cl.mostrar(img_dft, "DFT - " + SIN4_GIF)
    cl.mostrar(img_idft, "IDFT - " + SIN4_GIF)

def letra_i():
    imagem = cl.abrir("exercicio_fft/"+SINX3_GIF)
    img_dft = cl.dft_2D_gray_scale(imagem)
    img_idft = idft_2D_gray_scale(img_dft)

    cl.mostrar(imagem, SINX3_GIF)
    cl.mostrar(img_dft, "DFT - " + SIN4_GIF)
    cl.mostrar(img_idft, "IDFT - " + SIN4_GIF)

def letra_j():
    imagem = cl.abrir("exercicio_fft/"+SIN_ALL_GIF)
    img_dft = cl.dft_2D_gray_scale(imagem)
    img_idft = idft_2D_gray_scale(img_dft)

    cl.mostrar(imagem, SIN_ALL_GIF)
    cl.mostrar(img_dft, "DFT - " + SIN4_GIF)
    cl.mostrar(img_idft, "IDFT - " + SIN4_GIF)

def letra_k():
    imagem = cl.abrir("exercicio_fft/"+GAUSIAN_GIF)
    img_dft = cl.dft_2D_gray_scale(imagem)
    img_idft = idft_2D_gray_scale(img_dft)

    cl.mostrar(imagem, GAUSIAN_GIF)
    cl.mostrar(img_dft, "DFT - " + SIN4_GIF)
    cl.mostrar(img_idft, "IDFT - " + SIN4_GIF)

def letra_l():
    imagem = cl.abrir("exercicio_fft/"+SIN10_GAUSS_GIF)
    img_dft = cl.dft_2D_gray_scale(imagem)
    img_idft = idft_2D_gray_scale(img_dft)

    cl.mostrar(imagem, SIN10_GAUSS_GIF)
    cl.mostrar(img_dft, "DFT - " + SIN4_GIF)
    cl.mostrar(img_idft, "IDFT - " + SIN4_GIF)


letra_a()
