import numpy as np
from scipy import ndimage
import scipy
from scipy import signal
from numpy import exp

def check_module():
  return 'my name'

def boxBlur(im, k):
``` Return a  blured image filtered by box filter```


  return im_out

def convolve(im, kernel):
``` Return an image filtered by kernel```
  

  return im_out

def gradientMagnitude(im):
``` Return the sum of the absolute value of the graident  
    The gradient is the filtered image by Sobel filter ```


  return im_out

def horiGaussKernel(sigma, truncate=3):
  ```Return an one d kernel
  return some_array

def gaussianBlur(im, sigma, truncate=3):

  return gaussian_blured_image


def gauss2D(sigma=2, truncate=3):
  ```Return an 2-D array of gaussian kernel```

  return gaussian_kernel 

def unsharpenMask(im, sigma, truncate, strength):

  return sharpened_image

def bilateral(im, sigmaRange, sigmaDomain):

  return bilateral_filtered_image


def bilaYUV(im, sigmaRange, sigmaY, sigmaUV):
  ```6.865 only: filter YUV differently```
  return bilateral_filtered_image

# Helpers

