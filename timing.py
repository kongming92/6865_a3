import a3
import imageIO as io
import time

zebra = io.imread('zebra.png')
print 'Begin separable Gaussian'
start = time.time()
zebra1 = a3.gaussianBlur(zebra, 3, 3)
end = time.time()
print end - start, 'seconds'
print 'Begin 2D Gaussian'
start = time.time()
zebra2 = a3.convolve(zebra, a3.gauss2D(3, 3))
end = time.time()
print end - start, 'seconds'
io.imwrite(zebra1, 'zebraSeparableGaussian.png')
io.imwrite(zebra2, 'zebra2DGaussian.png')