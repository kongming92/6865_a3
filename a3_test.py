from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import a3
import matplotlib.cm as cm
import imageIO as myImageIO

## Test case ##
## Feel free to change the parameters or use the impulse as input

def test_box_blur():
  im=myImageIO.imread('pru.png')
  out=a3.boxBlur(im, 7)
  myImageIO.imwrite(out, 'test_boxBlur.png')

def test_convolve_gauss():
  im=myImageIO.imread('pru.png')
  gauss3=np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
  kernel=gauss3.astype(float)
  kernel=kernel/sum(sum(kernel))
  out=a3.convolve(im, kernel)
  myImageIO.imwrite(out, 'test_convolve.png')

def test_convolve_deriv():
  im=myImageIO.imread('pru.png')
  deriv=np.array([[-1, 1]])
  out=a3.convolve(im, deriv)
  myImageIO.imwrite(out, 'test_convolvDeriv.png')

def test_convolve_Sobel():
  im=myImageIO.imread('pru.png')
  Sobel=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  out=a3.convolve(im, Sobel)
  myImageIO.imwrite(out, 'test_sobel.png')

def test_grad():
  im=myImageIO.imread('pru.png')
  out=a3.gradientMagnitude(im)
  myImageIO.imwrite(out, 'test_magnitude.png')

def test_horigauss():
  im=myImageIO.imread('pru.png')
  kernel=a3.horiGaussKernel(2,3)
  out=a3.convolve(im, kernel)
  myImageIO.imwrite(out, 'test_horiGauss.png')

def test_gaussianBlur():
  im=myImageIO.imread('pru.png')
  out=a3.gaussianBlur(im, 2, 3)
  myImageIO.imwrite(out, 'test_gaussBlur.png')

def test_gauss2D():
  im=myImageIO.imread('pru.png')
  out=a3.convolve(im, a3.gauss2D())
  myImageIO.imwrite(out, 'test_gauss2DBlur.png')

def test_equal():
  im=myImageIO.imread('pru.png')
  out1=a3.convolve(im, a3.gauss2D())
  out2=a3.gaussianBlur(im,2, 3)
  res=abs(out1-out2);
  return (sum(res.flatten())<0.1)

def test_unsharpen():
  im=myImageIO.imread('zebra.png')
  out=a3.unsharpenMask(im, 1, 3, 1)
  myImageIO.imwrite(out, 'test_unsharpMask.png')

def test_bilateral():
  im=myImageIO.imread('lens-3-med.png', 1.0)
  out=a3.bilateral(im, 0.3, 1.4)
  myImageIO.imwrite(out, 'test_bilateral.png', 1.0)

def test_bilaYUV():
  im=myImageIO.imread('lens-3-small.png', 1.0)
  out=a3.bilaYUV(im, 0.3, 1.4, 6)
  myImageIO.imwrite(out, 'test_bilateralYUV.png', 1.0)

def impulse(h=100, w=100):
    out=myImageIO.constantIm(h, w, 0.0)
    out[h/2, w/2]=1
    return out


#Uncomment the following function to test your code


# test_box_blur()
# test_convolve_gauss()
# test_convolve_deriv()
# test_convolve_Sobel()
# test_grad()
# test_horigauss()
# test_gaussianBlur()
# test_gauss2D()
# print test_equal()

# test_unsharpen()
# test_bilateral()
test_bilaYUV()
