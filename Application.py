import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline

def dummy(val):
    pass

def renderPreview(img):
    if img.shape[1]*img.shape[0] >= 205000:
        ratio = img.shape[1]/img.shape[0]
        height = int(round((185000/ratio)**0.5, 0))
        width = int(round(height*ratio))
        return cv2.resize(img, (width,height))
    else:
        return img
    

def saveOutput(img, count, applyFilter=True):
    kernel = cv2.getTrackbarPos('Kernels', 'App')
    contrast = cv2.getTrackbarPos('Contrast', 'App')
    brightness = cv2.getTrackbarPos('Brightness', 'App')
    gamma = cv2.getTrackbarPos('Gamma', 'App')
    _filter = cv2.getTrackbarPos('Filters', 'App')
    if applyFilter:
        _filter = cv2.getTrackbarPos('Filters', 'App')
        filtered = filters[_filter](img)
    else:
        filtered = img
    modified = cv2.filter2D(filtered, -1, kernels[kernel])
    modified = cv2.convertScaleAbs(modified, alpha=contrast*0.04, beta=brightness-75)
    if gamma:
        modified = adjustGamma(modified, gamma=gamma*0.05)
    if _map:
        modified = cv2.applyColorMap(modified, _map-1)
    cv2.imwrite(f'output\output{count}_{fileName}', modified)

identityKernel = np.array([[0,0,0],[0,1,0],[0,0,0]])
sharpenKernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
boxBlur = np.array([[1/9, 1/9, 1/9],[1/9, 1/9, 1/9],[1/9, 1/9, 1/9]])
gaussianKernel1 = cv2.getGaussianKernel(7,0)
gaussianKernel2 = cv2.getGaussianKernel(9,0)
gaussianKernel3 = cv2.getGaussianKernel(11,0)
edgeDetector1 = np.array([[-1, -1, -1],[-1,  8, -1],[-1, -1, -1]])
edgeDetector2 = np.array([[1, 0, -1],[0, 0,  0],[-1,0,  1]])

def adjustGamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def _create_LUT_8UC1(x, y):
        spl = UnivariateSpline(x, y)
        return spl(range(256))
    
def warmingFilter(img):
        incrChLut = _create_LUT_8UC1([0, 64, 128, 192, 256],[0, 70, 140, 210, 256])
        
        decrChLut = _create_LUT_8UC1([0, 64, 128, 192, 256],[0, 30,  80, 120, 192])
        
        cB, cG, cR = cv2.split(img)
        cR = cv2.LUT(cR, incrChLut).astype(np.uint8)
        cB = cv2.LUT(cB, decrChLut).astype(np.uint8)
        img = cv2.merge((cB, cG, cR))
        cH, cS, cV = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        cS = cv2.LUT(cS, incrChLut).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((cH, cS, cV)), cv2.COLOR_HSV2RGB)
    
def coolingFilter(img):
        incrChLut = _create_LUT_8UC1([0, 64, 128, 192, 256],[0, 70, 140, 210, 256])
        
        decrChLut = _create_LUT_8UC1([0, 64, 128, 192, 256],[0, 30,  80, 120, 192])
        
        cB, cG, cR = cv2.split(img)
        cB = cv2.LUT(cB, incrChLut).astype(np.uint8)
        cR = cv2.LUT(cR, decrChLut).astype(np.uint8)
        img = cv2.merge((cB, cG, cR))
        cH, cS, cV = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        cS = cv2.LUT(cS, decrChLut).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((cH, cS, cV)), cv2.COLOR_HSV2RGB)

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

filters = [identity, warmingFilter, coolingFilter, enhanceDetails, pencilSketch,
           style, drawSketchBW, solidSketchBW, drawSketch, solidSketch]

fileName = 'trial5.jpg'
_colorOriginal = cv2.imread(fileName)
colorOriginal = renderPreview(_colorOriginal.copy())
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
cv2.createTrackbar('Gamma', 'App', 20, 100, dummy)
cv2.createTrackbar('Kernels', 'App', 0, len(kernels)-1, dummy)
cv2.createTrackbar('Filters', 'App', 0, len(filters)-1, dummy)
cv2.createTrackbar('Color', 'App', 0, 2, dummy)
cv2.createTrackbar('Color_Map', 'App', 0, 20, dummy)
count = 0
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
            try:
                saveOutput(img=_colorOriginal, count=count)
            except:
                print('Éxception')
                cv2.imwrite(f'output/output{count}_{fileName}', colorModified)
        elif colorScale == 1:
            try:
                temp = cv2.cvtColor(_colorOriginal, cv2.COLOR_BGR2GRAY)
                saveOutput(img=temp, count=count, applyFilter=False)
            except:
                print('Éxception')
                cv2.imwrite(f'output/output{count}_{fileName}', grayModified)
        else:
            try:
                temp = cv2.cvtColor(_colorOriginal, cv2.COLOR_BGR2HSV)
                saveOutput(img=temp, count=count)
            except:
                print('Éxception')
                cv2.imwrite(f'output/output{count}_{fileName}', colorModified)
    
    kernel = cv2.getTrackbarPos('Kernels', 'App')
    contrast = cv2.getTrackbarPos('Contrast', 'App')
    brightness = cv2.getTrackbarPos('Brightness', 'App')
    gamma = cv2.getTrackbarPos('Gamma', 'App')
    _filter = cv2.getTrackbarPos('Filters', 'App')
    _map = cv2.getTrackbarPos('Color_Map', 'App')
    
    if not p_filter == _filter:
            colorFilter = filters[_filter](colorOriginal)
            hsvFiltter = filters[_filter](hsvOriginal)
            
    colorModified = cv2.filter2D(colorFilter, -1, kernels[kernel])
    grayModified = cv2.filter2D(grayOriginal, -1, kernels[kernel])
    hsvModified = cv2.filter2D(hsvFiltter, -1, kernels[kernel])
    if gamma:
        colorModified = adjustGamma(colorModified, gamma=gamma*0.05)
        grayModified = adjustGamma(grayModified, gamma=gamma*0.05)
        hsvModified = adjustGamma(hsvModified, gamma=gamma*0.05)
    if _map:
        colorModified = cv2.applyColorMap(colorModified, _map-1)
        grayModified = cv2.applyColorMap(grayModified, _map-1)
        hsvModified = cv2.applyColorMap(hsvModified, _map-1)
    colorModified = cv2.convertScaleAbs(colorModified, alpha=contrast*0.04, beta=brightness-75)
    grayModified = cv2.convertScaleAbs(grayModified, alpha=contrast*0.04, beta=brightness-75)
    hsvModified = cv2.convertScaleAbs(hsvModified, alpha=contrast*0.04, beta=brightness-75)
    p_filter = _filter
    
cv2.destroyAllWindows()
