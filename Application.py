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

def identity(img):
    return img

def enhanceDetails(img):
    return cv2.detailEnhance(img)

def style(img):
    return cv2.stylization(img)

def pencilSketch(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (21, 21), 0, 0)
    imgBlend = cv2.divide(imgGray, imgBlur, scale=256)
    return cv2.cvtColor(imgBlend, cv2.COLOR_GRAY2BGR)

def solidSketchBW(img):
    return cv2.pencilSketch(img, sigma_s=80, sigma_r=0.08, shade_factor=0.05)[0]

def drawSketchBW(img):
    enhancedDetail = cv2.detailEnhance(img, sigma_s=15, sigma_r=0.55)
    edgePreserving1 = cv2.edgePreservingFilter(enhancedDetail, flags=1, sigma_s=60, sigma_r=0.4)
    return cv2.pencilSketch(edgePreserving1, sigma_s=60, sigma_r=0.07, shade_factor=0.05)[0]

def solidSketch(img):
    return cv2.pencilSketch(img, sigma_s=80, sigma_r=0.08, shade_factor=0.05)[1]
    
def drawSketch(img):
    enhancedDetail = cv2.detailEnhance(img, sigma_s=15, sigma_r=0.55)
    edgePreserving1 = cv2.edgePreservingFilter(enhancedDetail, flags=1, sigma_s=60, sigma_r=0.4)
    return cv2.pencilSketch(edgePreserving1, sigma_s=60, sigma_r=0.07, shade_factor=0.05)[1]


kernels = [identityKernel, sharpenKernel, boxBlur, gaussianKernel1, 
           gaussianKernel2, gaussianKernel3, edgeDetector1, edgeDetector2]

filters = [identity, enhanceDetails, style, pencilSketch, drawSketchBW, solidSketchBW, drawSketch, solidSketch]

fileName = 'test0.jpg'
colorOriginal = cv2.imread(fileName)
grayOriginal = cv2.cvtColor(colorOriginal, cv2.COLOR_BGR2GRAY)
hsvOriginal = cv2.cvtColor(colorOriginal, cv2.COLOR_BGR2HSV)
colorModified = colorOriginal.copy()
grayModified = grayOriginal.copy()
hsvModified = hsvOriginal.copy()
colorFilter = colorOriginal.copy()
hsvFilter = hsvOriginal.copy()

cv2.namedWindow('App', flags=4)
cv2.createTrackbar('Contrast', 'App', 25, 100, dummy)
cv2.createTrackbar('Brightness', 'App', 75, 150, dummy)
cv2.createTrackbar('Kernels', 'App', 0, len(kernels)-1, dummy)
cv2.createTrackbar('Filters', 'App', 0, len(filters)-1, dummy)
cv2.createTrackbar('Color', 'App', 0, 2, dummy)
count = 1
p_filter = 1
while 1:
    colorScale = cv2.getTrackbarPos('Color', 'App')
    if colorScale == 0:
        cv2.imshow('image', colorModified)
    elif colorScale == 1:
        cv2.imshow('image', grayModified)
    else:    
        cv2.imshow('image', hsvModified)
        
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
    
    kernel = cv2.getTrackbarPos('Kernels', 'App')
    contrast = cv2.getTrackbarPos('Contrast', 'App')
    brightness = cv2.getTrackbarPos('Brightness', 'App')
    _filter = cv2.getTrackbarPos('Filters', 'App')
    
    if not p_filter == _filter:
            colorFilter = filters[_filter](colorOriginal)
            hsvFiltter = filters[_filter](hsvOriginal)
            
    colorModified = cv2.filter2D(colorFilter, -1, kernels[kernel])
    grayModified = cv2.filter2D(grayOriginal, -1, kernels[kernel])
    hsvModified = cv2.filter2D(hsvFiltter, -1, kernels[kernel])
    colorModified = cv2.convertScaleAbs(colorModified, alpha=contrast*0.04, beta=brightness-75)
    grayModified = cv2.convertScaleAbs(grayModified, alpha=contrast*0.04, beta=brightness-75)
    hsvModified = cv2.convertScaleAbs(hsvModified, alpha=contrast*0.04, beta=brightness-75)
    p_filter = _filter
    
cv2.destroyAllWindows()
