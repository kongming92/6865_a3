import numpy as np
import imageIO as io
import math

def check_module():
    return 'Charles Liu'

def boxBlur(im, k):
    # Return a  blured image filtered by box filter
    shift = int(k/2)
    factor = 1.0 / (k**2)
    im_out = io.constantIm(im.shape[0], im.shape[1], 0)
    for y, x in imIter(im_out):
        for yp, xp in [(a, b) for a in xrange(k) for b in xrange(k)]:
            im_out[y, x] += getEdgePadded(im, y + yp - shift, x + xp - shift) * factor
    return im_out

def convolve(im, kernel):
    # Return an image filtered by kernel
    shiftY, shiftX = (int(kernel.shape[0] / 2), int(kernel.shape[1] / 2))
    im_out = io.constantIm(im.shape[0], im.shape[1], 0)
    for y, x in imIter(im_out):
        for yp, xp in imIter(kernel):
            im_out[y, x] += getEdgePadded(im, y + yp - shiftY, x + xp - shiftX) * kernel[yp, xp]
    return im_out

def gradientMagnitude(im):
    # Return the sum of the absolute value of the gradient
    # The gradient is the filtered image by Sobel filter
    s_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    s_Y = np.transpose(s_X)
    G_X = convolve(im, s_X)
    G_Y = convolve(im, s_Y)
    return (G_X**2 + G_Y**2) ** 0.5

def horiGaussKernel(sigma, truncate=3):
    # Return an one d kernel
    offset = int(math.ceil(truncate * sigma))
    kernel = [math.e ** -((i - offset)**2 / float(2 * sigma**2)) for i in xrange(2 * offset + 1)]
    return np.array([kernel]) / sum(kernel)

def gaussianBlur(im, sigma, truncate=3):
    horizKernel = horiGaussKernel(sigma, truncate)
    return convolve(convolve(im, horizKernel), np.transpose(horizKernel))

def gauss2D(sigma=2, truncate=3):
    # Return an 2-D array of gaussian kernel
    horizKernel = horiGaussKernel(sigma, truncate)
    return np.dot(np.transpose(horizKernel), horizKernel)

def unsharpenMask(im, sigma, truncate, strength):
    return im + strength * (im - gaussianBlur(im, sigma, truncate))

def bilateral(im, sigmaRange, sigmaDomain):
    truncate = 2
    offset = int(math.ceil(truncate * sigmaDomain))
    gaussianDistDomain = gauss2D(sigmaDomain, truncate)
    im_out = io.constantIm(im.shape[0], im.shape[1], 0)
    for y, x in imIter(im_out):
        k = 0
        s = np.array([0.0, 0.0, 0.0])
        for yp in xrange(y - offset, y + offset + 1):
            for xp in xrange(x - offset, x + offset + 1):
                # Get domain Gaussian
                domainGaussian = gaussianDistDomain[y - yp + offset, x - xp + offset]
                # Get range Gaussian
                imDist = np.sum((im[y, x] - getEdgePadded(im, yp, xp))**2) ** 0.5
                rangeGaussian = continuousGaussian(imDist, sigmaRange)
                # rangeGaussian = 1
                k += domainGaussian * rangeGaussian
                s += domainGaussian * rangeGaussian * getEdgePadded(im, yp, xp)
        im_out[y, x] = s / k

    return im_out

def bilaYUV(im, sigmaRange, sigmaY, sigmaUV):
    # 6.865 only: filter YUV differently
    imYUV = rgb2yuv(im)
    bilateralY = bilateral(imYUV, sigmaRange, sigmaY)
    bilateralUV = bilateral(imYUV, sigmaRange, sigmaUV)
    im_out = io.constantIm(im.shape[0], im.shape[1], 0)

    for y,x in imIter(im_out):
        im_out[y, x] = np.array([bilateralY[y, x, 0], bilateralUV[y, x, 1], bilateralUV[y, x, 2]])

    return yuv2rgb(im_out)

# Helpers

def continuousGaussian(x, sigma):
    return 1.0 / math.sqrt(2 * math.pi * sigma**2) * math.e ** -(float(x)**2 / (2 * sigma**2))

def imIter(im):
    for y in range(0,im.shape[0]):
        for x in range(0,im.shape[1]):
            yield (y,x)

def clipX(im, x):
    return min(im.shape[1] - 1, max(x, 0))

def clipY(im, y):
    return min(im.shape[0] - 1, max(y, 0))

def getEdgePadded(im, y, x):
    return im[clipY(im, y), clipX(im, x)];

def rgb2yuv(im):
    imyuv = im.copy()
    M = np.array([[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.1001]])
    (height, width, rgb) = np.shape(imyuv)
    for y in xrange(height):
        for x in xrange(width):
            imyuv[y][x] = np.dot(M, imyuv[y][x])
    return imyuv

def yuv2rgb(im):
    imrgb = im.copy()
    M = np.matrix([[1, 0, 1.13983], [1, -0.39465, -0.58060], [1, 2.03211, 0]])
    (height, width, yuv) = np.shape(imrgb)
    for y in xrange(height):
        for x in xrange(width):
            imrgb[y][x] = np.dot(M, imrgb[y][x])
    return imrgb
