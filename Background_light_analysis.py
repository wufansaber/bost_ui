import numpy as np
import os
import cv2
from skimage import morphology
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def defect_detection(im1):
    m=im1.shape[0]
    n=im1.shape[1]
    print(m,n)
    img = cv2.medianBlur(im1,5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU算法求阈值
    # cv2.imshow("binary", binary)
    # cv2.waitKey(0)
    #cv2.destroyAllWindows()
    '''分水岭---
    -------待补充-----'''
    cleaned = morphology.remove_small_objects(binary, min_size=10, connectivity=2)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    area_list1=[]
    n1=0
    for i in range(stats.shape[0]):
        if stats[i,0]!=0 and stats[i,1]!=0:
            area_list1.append(stats[i,4])
        else:
            n1+=1
    try:
        max_area=np.max(area_list1)
    except:
        max_area=''
    number=num_labels-n1
    return max_area,number


def backguround_ligt_result(filepath):
    #原图路径
    #filepath=r'\\10.0.32.20\RnD_Temp\wufan\64\Background_light_test\2022_02_09_11_00_14'
    #存图路径
    savepath=filepath
    #存图名称
    savename='底板.tiff'
    #起始tile，即拼图左上角的tile的编号
    tile1 = 1
    #X>=2，行数
    X = 37
    #Y>=2，列数
    Y = 14
    #整数，缩小2^shrink倍
    shrink = 4
    #图像排列，选择‘N’或‘U’
    pattern = 'N'
    #图像翻转
    flip = 0
    #边距像素数
    margin = 5
    #图像旋转角度
    theta = 0
    #拼图的每列视野数
    Ylim = 37

    l = round(2048/(2**shrink))
    merge = np.zeros((X*(l+margin),Y*(l+margin)))
    set = []
    max_area_list=[]
    number_list=[]
    R=filepath.split("_")[-3]
    cycle =filepath.split("_")[-1]
    if (tile1%Ylim)+Y-1>Ylim:
        Y = Ylim+1-(tile1%Ylim)
        print('Y out of boundary')
    for i in range(1,Y+1):
        for j in range(1,X+1):
                if len(set)>0:
                    set.append(set[-1]+1)
                else:
                    set.append(tile1)
    if pattern == 'U':
        T=np.empty(X*Y)
        for i in range(1,Y+1):
            if i%2:
                T[(i-1)*X:i*X] = set[(i-1)*X:i*X]
            else:
                T[(i-1)*X:i*X]= set[i*X-1:(i-1)*X-1:-1]
    elif  pattern == 'N':
        T = set
    tile1X = ((tile1%Ylim)-1)
    tile1Y = (tile1-(tile1%Ylim))/Ylim
    for i in range(1,X*Y+1):
        print(i)
        filename='wdi_tile%04d_01.tiff'%T[i-1]
        imgname=os.path.join(filepath,filename)
        if os.path.exists(imgname):
            img = mpimg.imread(imgname)
            img1 = cv2.imread(imgname)
            max_area,number=defect_detection(img1)
            if max_area!='':
                max_area_list.append(max_area)
            number_list.append(number)
        else:
            img = np.zeros((2048,2048))
        if flip==90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif flip==180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        else:
            pass
        if set[i-1]%Ylim!=0:
            x = int((l+margin)*((set[i-1]%Ylim)-1-tile1X) +1)
            y = int((l+margin)*((set[i-1]-(set[i-1]%Ylim))/Ylim-tile1Y) +1)
            print(x,y,l)
        else:
            x = int((l+margin)*(Ylim-1-tile1X) +1)
            y = int((l+margin)*((set[i-1]-Ylim)/Ylim-tile1Y) +1)
            print(x,y,l)
        img_shrink=cv2.resize(img,(int(2048*(0.5**shrink)),int(2048*(0.5**shrink))),interpolation=cv2.INTER_CUBIC)
        merge[x:x+l,y:y+l] = img_shrink
    sum_number=np.sum(number_list)
    maxarea=np.max(max_area_list)
    meanarea=np.mean(max_area_list)
    merge = np.asanyarray(merge, dtype="uint16")
    mean_merge=np.mean(merge[:,:])
    plt.figure()
    ax1=sns.heatmap(merge)
    #plt.show()
    plt.savefig(os.path.join(savepath,savename))
    plt.close()
    with open (os.path.join(savepath,'Background_light_test.txt'),'w') as fid:
        fid.write("最大亮域面积: \t %.4f\n"%maxarea)
        fid.write("平均值: \t %.4f\n"%mean_merge)
        fid.write("平均亮面积: \t %.4f\n"%meanarea)
        fid.write("亮区域个数: \t %.4f\n"%sum_number)

