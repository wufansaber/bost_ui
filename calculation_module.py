import cv2
import numpy as np
import pywt
from scipy import signal
from scipy.signal import argrelextrema


def image_16_8(image):
    min_16bit = np.min(image)
    max_16bit = np.max(image)
    # image_8bit = np.array(np.rint((255.0 * (image_16bit - min_16bit)) / float(max_16bit - min_16bit)), dtype=np.uint8)
    # 或者下面一种写法
    image_8bit = np.array(np.rint(255 * ((image - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    return image_8bit

def cal_light_Uniformity(img):
    img_mean = cv2.blur(img, (5, 5))
    img_resize_down = cv2.resize(img_mean, dsize=(100, 100),
                             interpolation=cv2.INTER_NEAREST)
    img_resize_up = cv2.resize(img_resize_down, dsize=(2048, 2048),
                             interpolation=cv2.INTER_LINEAR)
    maxindex =np.where(img_resize_up == np.max(img_resize_up))
    minindex= np.where(img_resize_up == np.min(img_resize_up))
    max_value=np.max(img_resize_down)
    min_value=np.min(img_resize_down)
    mean_value = np.mean(img_resize_down)
    Uniformity_max = 1 - (max_value - mean_value) / mean_value
    Uniformity_min = 1 - (mean_value - min_value) / mean_value
    Uniformity=min(Uniformity_max,Uniformity_min)*100

    return Uniformity,max_value,min_value,img_resize_down,img_resize_up

def cal_Modulation(center, rio_image):
    x = int(center[1] - 6)
    y = int(center[0] - 7)
    littie_rio_y = rio_image[x - 3:x + 3, y - 6:y + 6]

    y_mean = np.mean(littie_rio_y, 0)

    peak_h = signal.find_peaks(y_mean, distance=2)
    # peak_h=argrelextrema(y_mean,np.greater,order=3)
    fist = peak_h[0][0]
    last = peak_h[0][-1]
    new_y_mean = y_mean[fist:last]
    peak_l = argrelextrema(y_mean, np.less)
    a = peak_h[0]

    high_mean = np.mean(y_mean[peak_h[0]])
    low_mean = np.mean(y_mean[peak_l])
    Modulation_y = (high_mean - low_mean) / (high_mean + low_mean)


    x_col = int(center[1] + 8)
    y_col = int(center[0] + 6)
    littie_rio_x = rio_image[x_col - 6:x_col + 6, y_col - 2:y_col + 2]

    x_mean = np.mean(littie_rio_x, 1)

    peak_h = signal.find_peaks(x_mean, distance=2)
    fist = peak_h[0][0]
    last = peak_h[0][-1]
    new_x_mean = x_mean[fist:last]
    peak_l = argrelextrema(x_mean, np.less, order=2, mode='clip')

    high_mean = np.mean(x_mean[peak_h[0]])
    low_mean = np.mean(x_mean[peak_l])

    Modulation_x = (high_mean - low_mean) / (high_mean + low_mean)

    return Modulation_x, Modulation_y

def cal_center(rio_image):

    (h, w) = rio_image.shape[:2]  # 10
    center = (w // 2, h // 2)  # 1
    M = cv2.getRotationMatrix2D(center, 270, 1.0)  # 15
    img = cv2.warpAffine(rio_image, M, (w, h))  # 16
    img = cv2.flip(img, 1)

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    min_16bit = np.min(blur)
    max_16bit = np.max(blur)
    image_8bit = np.array(np.rint(255 * ((blur - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    dilation = cv2.dilate(image_8bit, kernel)
    # cv2.resizeWindow("image", 900, 900)
    # cv2.imshow('image',image_8bit)
    # cv2.imshow('dilation', dilation)
    # cv2.waitKey(0)
    ret3, th3 = cv2.threshold(dilation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nccomps = cv2.connectedComponentsWithStats(th3)
    labels = nccomps[1]
    status = nccomps[2]
    centroids = nccomps[3]
    for row in range(status.shape[0]):
        if status[row, :][2] < 55 and status[row, :][3] < 55 and status[row, :][2] > 30 and status[row, :][
            3] > 30 and status[row, :][4] > 1000:
            centroids_ture = centroids[row, :]
            return centroids_ture, img
        else:
            continue

def SML(imgsmall):
    I=imgsmall.astype(np.float)
    I=I/np.mean(I)
    t1=cv2.filter2D(I,-1,np.array( [[0,1,0],[1,-4,1],[0,1,0]] ))
    t2=cv2.filter2D(I,-1,np.array( [[1/3,1/3,1/3],[1/3,-8/3,1/3],[1/3,1/3,1/3]] ))
    t=np.abs(t1)+np.abs(t2)
    sml = np.mean(t)
    return sml

def cal_rota(image):
    image = image[100:1900, 1200 - 500:1200 + 500]
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    # min_16bit = np.min(blur)
    # max_16bit = np.max(blur)
    # image_8bit = np.array(np.rint(255 * ((blur - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    image_8bit = image_16_8(blur)
    ret3, binary_src = cv2.threshold(image_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_src = 255 - binary_src
    rows, cols = binary_src.shape
    scale = 40
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    eroded = cv2.erode(binary_src, kernel, iterations=1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=1)
    dilatedrow = dilatedrow / 255
    new = dilatedrow * image
    cv2.imwrite('new.png', new)
    num_peak = signal.find_peaks(new[800, :], distance=80)
    x = image.shape[0]  # 获取图像大小
    y = image.shape[1]
    get_center_j = np.zeros(shape=(x, 1))
    get_center = np.zeros(shape=(x, 1))
    mean_delta = 0

    for j in range(len(num_peak)):
        for i in range(x):
            # num_peak = signal.find_peaks(new[i, :], distance=80)
            le = num_peak[0][j] - 20
            if le < 0:
                le = 0
            re = num_peak[0][j] + 20
            if re > y:
                re = y
            a = new[i][le:re]
            get_center[i] = Weighted(a)
            # get_center[i] = num_peak[0][j] - 10 + center
        # max_center=np.max(get_center)
        # get_center_j = get_center/max_center+get_center_j
        up_coordinate = np.mean(get_center[0:50])
        down_coordinate = np.mean(get_center[-50:-1])
        delta = up_coordinate - down_coordinate
        mean_delta = mean_delta + delta
        # print(get_center)
    mean_delta = mean_delta / len(num_peak)
    return mean_delta

def Weighted( a):
    w = 0
    for i in range(len(a)):
        w = a[i] * (i + 1) + w
    center = w / np.sum(a)
    return center


def WAVV(imgsmall):
    I=imgsmall.astype(np.float)
    I = I/np.mean(I)
    coeffs2 = pywt.dwt2(I,'db6')
    ca,(ch,cv,cd) = coeffs2
    h2 = np.std(ch)
    v2 = np.std(cv)
    d2 = np.std(cd)
    fv = h2+v2+d2
    return fv

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))