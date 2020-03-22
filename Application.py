import cv2
import numpy as np

def dummy(val):
    pass


identityKernel = np.array([[0,0,0],[0,1,0],[0,0,0]])
sharpenKernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
boxBlur = np.array([[1/9, 1/9, 1/9],[1/9, 1/9, 1/9],[1/9, 1/9, 1/9]])
gaussianKernel1 = cv2.getGaussianKernel(7,0)
gaussianKernel2 = cv2.getGaussianKernel(9,0)
gaussianKernel3 = cv2.getGaussianKernel(11,0)
edgeDetector1 = np.array([[-1, -1, -1],[-1,  8, -1],[-1, -1, -1]])
edgeDetector2 = np.array([[1, 0, -1],[0, 0,  0],[-1,0,  1]])

kernels = [identityKernel, sharpenKernel, boxBlur, gaussianKernel1, 
           gaussianKernel2, gaussianKernel3, edgeDetector1, edgeDetector2]

fileName = 'test0.jpg'
colorOriginal = cv2.imread(fileName)
grayOriginal = cv2.cvtColor(colorOriginal, cv2.COLOR_BGR2GRAY)
hsvOriginal = cv2.cvtColor(colorOriginal, cv2.COLOR_BGR2HSV)
colorModified = colorOriginal.copy()
grayModified = grayOriginal.copy()
hsvModified = hsvOriginal.copy()

cv2.namedWindow('App')
cv2.createTrackbar('Contrast', 'App', 25, 100, dummy)
cv2.createTrackbar('Brightness', 'App', 75, 150, dummy)
cv2.createTrackbar('Filters', 'App', 0, len(kernels)-1, dummy)
cv2.createTrackbar('Color', 'App', 0, 2, dummy)

count = 1
while 1:
    colorScale = cv2.getTrackbarPos('Color', 'App')
    if colorScale == 0:
        cv2.imshow('App', colorModified)
    elif colorScale == 1:
        cv2.imshow('App', grayModified)
    else:    
        cv2.imshow('App', hsvModified)
        
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('o'):
        cv2.imshow('Original', colorOriginal)
    elif k == ord('s'):
        count+=1
        if colorScale == 0:
            cv2.imwrite(f'output{count}_{fileName}', colorModified)
        elif colorScale == 1:
            cv2.imwrite(f'output{count}_{fileName}', grayModified)
        else:
            cv2.imwrite(f'output{count}_{fileName}', hsvModified)
    
    kernel = cv2.getTrackbarPos('Filters', 'App')
    contrast = cv2.getTrackbarPos('Contrast', 'App')
    brightness = cv2.getTrackbarPos('Brightness', 'App')
    
    colorModified = cv2.filter2D(colorOriginal, -1, kernels[kernel])
    grayModified = cv2.filter2D(grayOriginal, -1, kernels[kernel])
    hsvModified = cv2.filter2D(hsvOriginal, -1, kernels[kernel])
    colorModified = cv2.convertScaleAbs(colorModified, alpha=contrast*0.04, beta=brightness-75)
    grayModfied = cv2.convertScaleAbs(grayModified, alpha=contrast*0.04, beta=brightness-75)
    hsvModified = cv2.convertScaleAbs(hsvModified, alpha=contrast*0.04, beta=brightness-75)
    
cv2.destroyAllWindows()
