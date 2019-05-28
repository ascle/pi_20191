import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from PIL import Image

IMG_LENNA_JPG = '/home/aluno/Imagens/lena.jpeg'
IMG_LENNA_PNG = '/home/aluno/Imagens/lena.png'

# usar sempre o dtype int8
def abrir(path):
    return np.asarray(Image.open(path))

# se ligar no grayscale (color map), imagens pequenas (interpolação nearest)
def mostrar(img):
    plt.imshow(img)
    plt.show()

def ncanais(img)    :
    return img.shape[2]

