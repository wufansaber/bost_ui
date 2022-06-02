import os
import sys

import cv2
import numpy as np
from PyQt5 import Qt
from PyQt5.QtCore import QRectF, QSizeF, QPointF
from PyQt5.QtGui import QImage, QPainter, QPixmap, QColor
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene
from matplotlib import pyplot as plt, cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import bost_UI
from bost_UI import  Ui_MainWindow
import common2 as com
import DDS2 as DDS
import time
import calculation_module
global C,Xoffset,Yoffset
from scipy import signal
from scipy.signal import argrelextrema


    def show_image(self, image, scale=0.438):
        x = image.shape[1]  # 获取图像大小
        y = image.shape[0]
        self.zoomscale = float(scale)  # 图片放缩尺度

        image_8bit = np.array(image)
        frame = QImage(image_8bit, x, y, QImage.Format_Grayscale8)
        self.pix = QPixmap.fromImage(frame)

        self.item = QGraphicsPixmapItem(self.pix)  # 创建像素图元
        self.item.setScale(self.zoomscale)
        self.item.setFlags(QGraphicsPixmapItem.ItemIsFocusable |
                           QGraphicsPixmapItem.ItemIsMovable)
        self.item.update()
        self.setSceneDims()
        self.graphicsView.setRenderHints(QPainter.Antialiasing | QPainter.HighQualityAntialiasing |
                                         QPainter.SmoothPixmapTransform)
        self.graphicsView.setViewportUpdateMode(self.graphicsView.SmartViewportUpdate)
        self.scene = QGraphicsScene()  # 创建场景s
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)

    def start(self):
        self.z_read.start()

    def work(self):
        # 计时器每秒计数

        self.timer.start(30)
        # 计时开始
        self.workThread.start()
        # 当获得循环完毕的信号时，停止计数
        # self.workThread.save_image_Signal.connect(self.timer.stop)

    def work2(self):
        # 计时器每秒计数
        self.timer2.start(30)
        # 计时开始
        self.workThread2.start()


    def start_camer(self):
        self.save_image_act()


    def start_nine(self):
        self.nine_View()


    def stop_camer(self):
        self.timer.stop()
        self.timer2.stop()

    def get_z_display(self):
        zData = C.sendandrecv({"CCP": "WDI GET 4 TIMEOUT 1000"})
        # print('wdifocuse=%fum' % zData[b'data'][0])
        z = format(zData[b'data'][0], '.3f')
        self.z_display.setText(z)



    def diplay_z_value(self, str):
        self.z_display.setText(str)


    def get_x_display(self):
        xData = C.sendandrecv({"CCP": "SERVO01 GET 9 TIMEOUT 5000"})
        x = format(xData[b'data'][0], '.3f')
        self.x_display.setText(x)


    def get_y_display(self):
        yData = C.sendandrecv({"CCP": "SERVO02 GET 9 TIMEOUT 5000"})
        y = format(yData[b'data'][0], '.3f')
        self.y_display.setText(y)



    def init_ds(self):
        outputfolder = com.setOutputFolder('output/vibration')
        logger = com.createLogger(outputfolder)
        com.logger = logger
        com.outputfolder = outputfolder
        com.tilemap = com.TileMap('save/TM518.txt')
        readTMdefine = com.readTMdefine
        C = DDS.NSDS(outputfolder)
        # ip = '10.0.32.101'
        ip = '127.0.0.1'
        C.init_msgclient(ip, '6666')
        C.init_datreceiver(ip, '7777')
        C.init_imgreceiver(ip, '20183')
        com.C = C
        C._init_LED()
        C.MSG = {}
        dct_tm, dct_ij, dct_param = readTMdefine('save/TM518.txt')
        Xoffset = dct_param['XOffset']
        Yoffset = dct_param['YOffset']

        return C, outputfolder, logger, Xoffset, Yoffset


    def chip_in(self):
        # Chip In
        C.sendandrecv({"CCP": "MOTOR_C01 MOV 2"})
        time.sleep(5)
        C.sendandrecv({"CCP": "MOTOR_C02 MOV 2"})

        self.get_y_display()
        self.get_x_display()


    def chip_out(self):
        # Chip out
        C.sendandrecv({"CCP": "SERVO01 RESET"})
        C.sendandrecv({"CCP": "SERVO02 RESET"})
        time.sleep(10)
        C.sendandrecv({"CCP": "MOTOR_C02 MOV 1"})
        time.sleep(15)
        C.sendandrecv({"CCP": "MOTOR_C01 MOV 1"})


    def x_move_resets(self):
        C.sendandrecv({"CCP": "SERVO01 RESET"})
        self.get_x_display()

    def y_move_resets(self):
        C.sendandrecv({"CCP": "SERVO02 RESET"})
        self.get_y_display()


    # def move_x(self,x_tid):
    #     x_tid=int(x_tid)
    #     if x_tid<=0:
    #         x_tid=1
    #     if x_tid>14:
    #         x_tid=14
    #     tid=(int(x_tid)-1)*37+1
    #     tilemap = com.TileMap('save/TM518.txt')
    #     xy = tilemap.t2xy([tid])[0]
    #     x = xy[0]
    #     C.sendandrecv({"CCP": "SERVO01 MOV 0 %.2f 0" % (x)})
    #     self.get_x_display()

    def move_x(self, x):
        x = float(x)
        C.sendandrecv({"CCP": "SERVO01 MOV 0 %.2f 0" % (x)})

        self.get_x_display()


    def move_x_step(self, mark):
        a = self.x_display.text()
        b = self.x_setp_value.text()

        if mark == 1:
            step = float(a) + float(b)
        else:
            step = float(a) - float(b)
        C.sendandrecv({"CCP": "SERVO01 MOV 0 %.2f 0" % (step)})
        self.get_x_display()


    def move_y_step(self, mark):
        a = self.y_display.text()
        b = self.y_setp_value.text()
        if mark == 1:
            step = float(a) + float(b)
        else:
            step = float(a) - float(b)
        C.sendandrecv({"CCP": "SERVO02 MOV 0 %.2f 0" % (step)})
        self.get_y_display()


    def move_z_step(self, mark):
        a = self.z_display.text()
        b = self.z_setp_value.text()
        if mark == 1:
            step = float(a) + float(b)
        else:
            step = float(a) - float(b)
        C.sendandrecv({"CCP": "WDI SET 4 %f TIMEOUT 1000" % step})
        self.get_z_display()


    def move_y(self, y):
        y = float(y)
        C.sendandrecv({"CCP": "SERVO02 MOV 0 %.2f 0" % (y)})
        self.get_y_display()


    def move_z(self, z):
        z = float(z)
        C.sendandrecv({"CCP": "WDI SET 4 %f TIMEOUT 1000" % z})
        self.get_z_display()


    def Brightfield_on_act(self):
        greenLEDcurrent = float(self.Brightfield_value.text())
        C.sendandrecv({"msgID": 1, "CCP": "LED_G SET 1 %.3f" % greenLEDcurrent})
        C.sendandrecv({"CCP": "LED_G OPEN"})


    def Brightfield_off_act(self):
        C.sendandrecv({"CCP": "LED_G CLOSE"})


    def fluorsecent_on(self):
        blueLEDcurrent = float(self.fluorsecent_value.text())
        C.sendandrecv({"msgID": 1, "CCP": "LED_B SET 1 %.3f" % blueLEDcurrent})
        C.sendandrecv({"CCP": "LED_B OPEN"})


    def fluorsecent_off(self):
        C.sendandrecv({"CCP": "LED_B CLOSE"})


    def sig_image_act(self):
        expTime = float(self.exposure_time.text())

        localtime0 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        localfolder = com.setOutputFolder('output/')

        output_img = os.path.join(localfolder, '%s.tiff' % (localtime0))
        image = self.cap_image(expTime, 1, output_img)
        image = calculation_module.image_16_8(image)
        self.show_image(image)


    def nine_View(self):
        expTime = float(self.exposure_time.text())
        image = self.cap_image(expTime, 0)
        # image = self.image_16_8(image)
        image_mid = calculation_module.image_16_8(image[924:1124, 924:1124])
        image_midup = calculation_module.image_16_8(image[0:200, 924:1124])
        image_middown = calculation_module.image_16_8(image[1838:2038, 924:1124])
        image_midleft = calculation_module.image_16_8(image[924:1124, 0:200])
        image_midright = calculation_module.image_16_8(image[924:1124, 1838:2038])
        image_upright = calculation_module.image_16_8(image[0:200, 1838:2038])
        image_upleft = calculation_module.image_16_8(image[0:200, 0:200])
        image_downleft = calculation_module.image_16_8(image[1838:2038, 0:200])
        image_downright = calculation_module.image_16_8(image[1838:2038, 1838:2038])
        nine_view_image = np.zeros((620, 620))
        nine_view_image[0:200, 0:200] = image_upleft
        nine_view_image[0:200, 210:410] = image_midup
        nine_view_image[0:200, 420:620] = image_upright
        nine_view_image[210:410, 0:200] = image_midleft
        nine_view_image[210:410, 210:410] = image_mid
        nine_view_image[210:410, 420:620] = image_midright
        nine_view_image[420:620, 0:200] = image_downleft
        nine_view_image[420:620, 210:410] = image_middown
        nine_view_image[420:620, 420:620] = image_downright
        # nine_view_image = self.image_16_8(nine_view_image)
        nine_view_image = np.array(nine_view_image, dtype=np.uint8)
        # cv2.imwrite('123.tif',nine_view_image)
        self.show_image(nine_view_image, 1)


    def save_image_act(self):
        expTime = float(self.exposure_time.text())
        image = self.cap_image(expTime, 0)
        image = calculation_module.image_16_8(image)
        self.show_image(image)
        # outputfolder = self.com.setOutputFolder('output\BOST_FOR_FCOUSE')
        # cv2.imwrite(outputfolder + r'\test.tiff', image)


    def fouce_off(self):
        C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 0 TIMEOUT 1000"})

        self.z_read.terminate()


    def fouce_on(self):
        C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 1 TIMEOUT 1000"})
        basic_control.start()

    def make_0_start(self):
        C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 0 TIMEOUT 1000"})
        print('AutoFocus Off!')
        C.sendandrecv({"CCP": "WDI DEVICECONTROL 3 TIMEOUT 5000"})
        print('Make0 OK!')


    def setBackground(self, color):
        """设置背景颜色
        :param color: 背景颜色
        :type color: QColor or str or GlobalColor
        """
        if isinstance(color, QColor):
            self.graphicsView.setBackgroundBrush(color)
        elif isinstance(color, (str, Qt.GlobalColor)):
            color = QColor(color)
            if color.isValid():
                self.graphicsView.setBackgroundBrush(color)

    def setPixmap(self, pixmap, fitIn=True):
        """加载图片
        :param pixmap: 图片或者图片路径
        :param fitIn: 是否适应
        :type pixmap: QPixmap or QImage or str
        :type fitIn: bool
        """

        frame = QImage(pixmap,2048,2048, QImage.Format_Grayscale8)
        self.pixmap = QPixmap.fromImage(frame)
        self._item.setPixmap(self.pixmap)
        self._item.update()
        self.setSceneDims()
        if fitIn:
            self.fitInView(QRectF(self._item.pos(), QSizeF(
                self.pixmap.size())), Qt.KeepAspectRatio)
        self._item.update()

    def setSceneDims(self):
        if not self.pix:
            return
        self.graphicsView.setSceneRect(QRectF(QPointF(0, 0), QPointF(self.pix.width(), self.pix.height())))

        print(self.pix.width())
    def fitInView(self, rect, flags=Qt.IgnoreAspectRatio):


        unity = self.graphicsView.transform().mapRect(QRectF(0, 0, 1, 1))
        self.graphicsView.scale(1 / unity.width(), 1 / unity.height())
        viewRect = self.graphicsView.viewport().rect()
        sceneRect = self.graphicsView.transform().mapRect(rect)
        x_ratio = viewRect.width() / sceneRect.width()
        y_ratio = viewRect.height() / sceneRect.height()
        if flags == Qt.KeepAspectRatio:
            x_ratio = y_ratio = min(x_ratio, y_ratio)
        elif flags == Qt.KeepAspectRatioByExpanding:
            x_ratio = y_ratio = max(x_ratio, y_ratio)
        self.graphicsView.scale(x_ratio, y_ratio)
        self.graphicsView.centerOn(rect.center())




    def cap_image(self,expTime,save,outputfolder=None):

        C.sendandrecv({"CCP": "CAM SET 2 %.3f" % expTime})
        C.sendandrecv({"CCP": "CAM SET 3 0 0 2048 2048"})
        C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 1 0 0"})
        data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
        img = np.reshape(np.frombuffer(data[b'data'], np.uint16), [2048, 2048])

        (h, w) = img.shape[:2]  # 10
        center = (w // 2, h // 2)  # 1
        M = cv2.getRotationMatrix2D(center, -90, 1.0)  # 15
        img = cv2.warpAffine(img, M, (w, h))  # 16
        img = cv2.flip(img, 1)
        if save==1:
            cv2.imwrite(outputfolder,img)
        return  img

    def close_all(self):
        C.sendandrecv({"CCP": "LED_B CLOSE"})
        C.sendandrecv({"CCP": "LED_G CLOSE"})
        C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 0 TIMEOUT 1000"})
        sys.exit(0)



    def measure_shading(self):
        blueLEDcurrent = float(self.fluorsecent_value.text())
        C.sendandrecv({"msgID": 1, "CCP": "LED_B SET 1 %.1f" % blueLEDcurrent})
        C.sendandrecv({"CCP": "LED_B OPEN"})
        localtime0 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        name = localtime0
        localfolder = com.setOutputFolder(os.path.join('output\shading' + '/' + name))
        output_shading_plot = os.path.join(localfolder, 'shading_plot.png' )
        output_img = os.path.join(localfolder, 'shading.tiff' )
        expTime = float(self.exposure_time.text())
        img=basic_control.cap_image(expTime,1,output_img)
        C.sendandrecv({"CCP": "LED_B CLOSE"})
        Uniformity,max_value,min_value,img_resize_down,img_resize_up=self.cal_light_Uniformity(img)
        plt=self.polt_3d(img_resize_down)
        plt.savefig(os.path.join(output_shading_plot))
        img_show = calculation_module.image_16_8(img_resize_up)
        basic_control.show_image(img_show)

    #       f = open(os.path.join(localfolder, 'shading.txt'), 'a')
    #       f.write('maxindex=%d %d \nmax_value=%.3f\nmin_value=%.3f\nmean_value=%.3f\nUniformity=%.3f\n' % (maxindex[0][0],maxindex[0][1],max_value, min_value, mean_value,Uniformity))
    #       f.close()

    def cal_light_Uniformity(self,img):
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



    def polt_3d(self,img):
        # 准备数据
        sp = img.shape
        h = int(sp[0])  # height(rows) of image
        w = int(sp[1])  # width(colums) of image
        fig = plt.figure(figsize=(16, 12))
        ax = fig.gca(projection="3d")
        imgd = np.array(img)
        x = np.arange(0, w, 1)
        y = np.arange(0, h, 1)
        x, y = np.meshgrid(x, y)
        z = imgd
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm)  # cmap指color map
        # 自定义z轴
        maxz=np.max(z)
        minz=np.min(z)
        ax.set_zlim(minz, maxz)
        ax.zaxis.set_major_locator(LinearLocator(10))  # z轴网格线的疏密，刻度的疏密，20表示刻度的个数
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))  # 将z的value字符串转为float，保留2位小数
        # 设置坐标轴的label和标题
        ax.set_xlabel('x', size=15)
        ax.set_ylabel('y', size=15)
        ax.set_zlabel('z', size=15)
        ax.set_title("Surface plot", weight='bold', size=20)
        # 添加右侧的色卡条
        fig.colorbar(surf, shrink=0.6, aspect=8)  # shrink表示整体收缩比例，aspect仅对bar的宽度有影响，aspect值越大，bar越窄
        plt.show()
        plt.savefig(os.path.join('output\shading\shading_plot.png' ))
        return  plt



    def Background_light_test(self):
        x = Xoffset + 17.98 / 2
        y = Yoffset + 48.6 / 2
        C.sendandrecv({"CCP": "SERVO01 MOV 0 %.2f 0" % (x)})
        C.sendandrecv({"CCP": "SERVO02 MOV 0 %.2f 0" % (y)})
        outputfolder=com.setOutputFolder('output\Background_light_test')
        output_img = os.path.join(outputfolder, 'no_light.tiff' )
        expTime = 0.03
        img=basic_control.cap_image(expTime,1,output_img)
        mean_img=np.mean(img)
        max=np.max(img)
        min=np.min(img)
        f = open(os.path.join(outputfolder, 'no_light.txt'), 'a')
        f.write('mean=%.3f\tmax=%.3f\n' % (mean_img,max))
        f.close()
        self.scan_image()


    def scan_image(self):
        # 扫描对焦
        global tilenumber, move_localfolder
        cyc = 1
        FMfile = 'save/FM518_S.txt'
        FM, FMZ = com.readFocusMap(FMfile)
        tilenumber = len(FM)
        tilemap = com.TileMap('save/TM518.txt')
        expTime = 0.03
        outputfolder = com.setOutputFolder('output/Background_light_test')
        blueLEDcurrent=10
        C.sendandrecv({"CCP": "CAM SET 2 %.3f" % expTime})

        C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 1 TIMEOUT 1000"})  # AutoFocus On!
        C.sendandrecv({"CCP": "SERVO01 MOV 4 %.2f 0"})
        C.sendandrecv({"CCP": "SERVO02 MOV 4 %.2f 0"})
        wait_time = float(0.1)
        time.sleep(wait_time)
        fv_tile = []
        zfocus_wdi_tile = []
        localtime0 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        move_localfolder = com.setOutputFolder('output\Background_light_test\\' + localtime0)
        C.sendandrecv({"msgID": 1, "CCP": "LED_B SET 1 %.1f" % blueLEDcurrent})
        #C.sendandrecv({"CCP": "LED_G OPEN"})
        C.sendandrecv({"CCP": "LED_B OPEN"})
        for i in range(1, cyc + 1):
            for tid in FM:
                # print(tid)
                xy = tilemap.t2xy([tid])[0]
                x = xy[0]
                y = xy[1]
                C.sendandrecv({"CCP": "SERVO01 MOV 0 %.2f 0" % (x)})
                C.sendandrecv({"CCP": "SERVO02 MOV 0 %.2f 0" % (y)})
                C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 1 TIMEOUT 1000"})  # AutoFocus On!
                time.sleep(wait_time)
                C.sendandrecv({"CCP": "WDI GET 5 TIMEOUT 1000"})  # To Wait for FocusOver and Get z
                C.sendandrecv({"CCP": "CAM SET 3 0 0 2048 2048"})
                C.sendandrecv({"CCP": "LED_B OPEN"})
                C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 0 0 0"})
                data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
                C.sendandrecv({"CCP": "LED_B CLOSE"})
                wdi_img = com.data2image(data,[2048,2048])
                output_img = os.path.join(move_localfolder, 'wdi_tile%04d_%02d.tiff' % (tid, i))
                cv2.imwrite(output_img, wdi_img)

        C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 0 TIMEOUT 1000"})  # AutoFocus Off!

        def z_get_imge(self):

            zData = C.sendandrecv({"CCP": "WDI GET 4 TIMEOUT 1000"})
            # print('wdifocuse=%fum' % zData[b'data'][0])
            z = format(zData[b'data'][0], '.3f')
            zfocus=float(z)
            greenLEDcurrent = float(self.Brightfield_value.text())
            blueLEDcurrent  = float(self.fluorsecent_value.text())
            C.sendandrecv({"msgID": 1, "CCP": "LED_G SET 1 %.1f" % greenLEDcurrent})
            C.sendandrecv({"msgID": 1, "CCP": "LED_B SET 1 %.1f" % blueLEDcurrent})
            C.sendandrecv({"CCP": "CAM SET 3 0 0 2048 2048"})

            zRange = 8
            zStep = 0.2  # um
            localtime0 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            name=localtime0
            outputfolder='output\z_get_imge'
            localfolder = com.setOutputFolder(os.path.join('output\z_get_imge' + '/' + name))
            file = os.path.join(localfolder, name + '.txt')
            f = open(file, 'a')

            print('Start test\tAt:%s\tzRange=%f\tzStep=%f' % (localtime0, zRange, zStep))
            f.write('Start test\tAt:%s\tzRange=%f\tzStep=%f\n' % (localtime0, zRange, zStep))
            f.close()
            expTime = float(self.exposure_time.text())
            for cycle in range(1):
                f = open(file, 'a')
                localtime = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
                print('cycle:%d\tAt:%s' % (cycle, localtime))
                f.write('cycle:%d\tAt:%s\n' % (cycle, localtime))
                f.close()


                with open(file, 'a') as f:


                    imgpath = localfolder + '/img_cycle%02d' % (cycle)
                    if not os.path.exists(imgpath):
                        os.mkdir(imgpath)
                    C.sendandrecv({"CCP": "CAM SET 3 0 0 2048 2048"})
                    print('zfocus:', zfocus)
                    C.sendandrecv({"CCP": "WDI SET 4 %f TIMEOUT 1000" % zfocus})  # 运动到z
                    print('cycle:%02d \tzfocus:%.2f' % (cycle, zfocus))
                    f.write('cycle:%02d\tzfocus:%.2f\n' % (cycle,  zfocus))

                    C.sendandrecv({"CCP": "CAM SET 2 0.002"})
                    C.sendandrecv({"CCP": "LED_G OPEN"})
                    C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 0 0 0"})  # 返回焦点数
                    data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
                    img = com.data2image(data)
                    cv2.imwrite(
                        os.path.join(localfolder, 'img_cycle_%02d-BF_z%.2fum.tiff' % ( cycle, zfocus)), img)
                    C.sendandrecv({"CCP": "LED_G CLOSE"})
                    C.sendandrecv({"CCP": "CAM SET 2 %.3f" % expTime})
                    C.sendandrecv({"CCP": "LED_B OPEN"})
                    C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 0 0 0"})
                    #             data = C.sendandrecv({"CCP": "CAM_CAPTURER TRIGGERPHOTO 0 0 0"})
                    data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
                    img = com.data2image(data)
                    cv2.imwrite(
                        os.path.join(localfolder, 'img_cycle_%02d-FF_z%.2fum.tiff' % ( cycle, zfocus)), img)
                    C.sendandrecv({"CCP": "LED_B CLOSE"})
                    time.sleep(0.02)

                    f.write('z\t G_WAV\t G_SML\t B_WAV\t B_SML\n')
                    C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 0 TIMEOUT 1000"})
                    print('AutoFocus Off!')

                    for z in np.arange(zfocus - zRange, zfocus + zRange + zStep, zStep):
                        C.sendandrecv({"CCP": "WDI SET 4 %f TIMEOUT 1000" % z})  # 运动到z
                        time.sleep(0.3)
                        C.sendandrecv({"CCP": "CAM SET 2 0.002"})
                        C.sendandrecv({"CCP": "LED_G OPEN"})
                        C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 0 0 0"})

                        data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
                        img = com.data2image(data)
                        C.sendandrecv({"CCP": "LED_G CLOSE"})
                        imgsmall = img[900:1100, 900:1100]
                        fvG_WT = calculation_module.WAVV(imgsmall)  # 反应照片清晰度的对焦值，小波变换
                        fvG_ML = calculation_module.SML(imgsmall)  # 反应照片清晰度的对焦值，改进的拉普拉斯算子

                        cv2.imwrite(os.path.join(imgpath, 'BFimg_cycle_%02d_z%.2fum_fv%.3f.tiff' % (
                         cycle, z, fvG_WT)), img)

                        C.sendandrecv({"CCP": "CAM SET 2 0.03"})
                        C.sendandrecv({"CCP": "LED_B OPEN"})
                        C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 0 0 0"})
                        #                 data = C.sendandrecv({"CCP": "CAM_CAPTURER TRIGGERPHOTO 0 0 0"})
                        data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
                        img = com.data2image(data)
                        C.sendandrecv({"CCP": "LED_B CLOSE"})
                        imgsmall = img[900:1100, 900:1100]
                        fvB_WT = calculation_module.WAVV(imgsmall)  # 反应照片清晰度的对焦值，小波变换
                        fvB_ML = calculation_module.SML(imgsmall)  # 反应照片清晰度的对焦值，改进的拉普拉斯算子
                        #                 print('pdB=%.3f\t fvB_WT=%.3f\t fvB_ML=%.3f' %(data[b'data'][0],fvB_WT,fvB_ML))
                        cv2.imwrite(os.path.join(imgpath, 'FFimg_cycle_%02d_z%.2fum_fv%.3f.tiff' % (
                        cycle, z, fvB_ML)), img)

                        time.sleep(0.02)
                        print('z=%.3f\t G_WAV=%.3f\t G_SML=%.3f\t B_WAV=%.3f\t B_SML=%.3f' % (
                        z, fvG_WT, fvG_ML, fvB_WT, fvB_ML))
                        f.write('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' % (z, fvG_WT, fvG_ML, fvB_WT, fvB_ML))

            f = open(file, 'a')
            print(('cycle:%d finish!') % cycle)
            f.write(('cycle:%d finish!\n') % cycle)
            f.close()

            f = open(file, 'a')
            localtime = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
            print('Finish!\tAt:%s' % (localtime))
            f.write('Finish!\tAt:%s\n' % (localtime))
            f.close()

    def static_focus(self):
        move_z = 10
        C.sendandrecv({"CCP": "CAM SET 3 1024 1024 256 256"})
        localtime0 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        localfolder = com.setOutputFolder('output\static_focus/' + localtime0)
        name = 'static_focus'
        file = os.path.join(localfolder, name + '.txt')
        f = open(file, 'a')
        # with open(os.path.join(localfolder, 'focustest.txt'),'a') as f:

        print('%s \tStart test\tAt:%s' % (name, localtime0))
        f.write('%s \t movez:%d \tStart test\tAt:%s\n\n' % (name, move_z, localtime0))
        f.close()
        FMfile = 'save/FM4x7.txt'
        FM, FMZ = com.readFocusMap(FMfile)

        tilemap = com.TileMap('save/TM518.txt')
        for tid in FM:
            # print(tid)
            xy = tilemap.t2xy([tid])[0]
            x = xy[0]
            y = xy[1]
            C.sendandrecv({"CCP": "SERVO01 MOV 0 %.2f 0" % (x)})
            C.sendandrecv({"CCP": "SERVO02 MOV 0 %.2f 0" % (y)})
            C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 1 TIMEOUT 1000"})  # AutoFocus On!
            time.sleep(1)
            C.sendandrecv({"msgID": 1, "CCP": "LED_G SET 1 %.3f" % 0.3})
            C.sendandrecv({"CCP": "LED_G OPEN"})
            repeat_times = 10
            zfocus_up_all = []
            zfocus_down_all = []
            fv_up_all = []
            fv_down_all = []

            for i in range(1, repeat_times + 1):
                with open(file, 'a') as f:
                    C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 1 TIMEOUT 1000"})  # AutoFocus On!
                    zData = C.sendandrecv({"CCP": "WDI GET 5 TIMEOUT 1000"})  # To Wait for FocusOver and Get z
                    zData_up = float(zData[b'data'][0])
                    time.sleep(1)
                    C.sendandrecv({"CCP": "CAM SET 2 %.3f" % 0.004})
                    C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 0 0 0"})
                    data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
                    img = com.data2image(data,[256,256])
                    output_img = os.path.join(localfolder, 'tile_%03d_up_%03d.tiff' % (tid,i))
                    cv2.imwrite(output_img, img)
                    fv_midd_up = calculation_module.WAVV(img)
                    zfocus_up_all = np.append(zfocus_up_all, zData_up)
                    fv_up_all = np.append(fv_up_all, fv_midd_up)
                    C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 0 TIMEOUT 1000"})  # AutoFocus Off!
                    C.sendandrecv({"CCP": "WDI SET 4 %f TIMEOUT 1000" % (zData_up + move_z)})
                    time.sleep(1)
                    f.write('tid：%d from up: cyc:%d \tzfocus:%.3f \tfv_value:%.3f \n' % (tid,i, zData_up, fv_midd_up))
            uniformity=max(zfocus_up_all)-min(zfocus_up_all)
            with open(file, 'a') as f:
                f.write('tid：%d from up\tuniformity:%.3f \n' % (tid, uniformity))
            for i in range(1, repeat_times + 1):
                with open(file, 'a') as f:
                    C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 1 TIMEOUT 1000"})  # AutoFocus On!
                    zData = C.sendandrecv({"CCP": "WDI GET 5 TIMEOUT 1000"})  # To Wait for FocusOver and Get z
                    zData_down = float(zData[b'data'][0])
                    # time.sleep(0.1)
                    C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 0 0 0"})
                    data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
                    img = com.data2image(data,[256,256])
                    output_img = os.path.join(localfolder, 'tile_%03d_down_%03d.tiff' % (tid,i))
                    cv2.imwrite(output_img, img)
                    # fv_upleft=WAVV(img[20:220,20:220])
                    # fv_upright=WAVV(img[20:220,1828:2028])
                    fv_midd_down = calculation_module.WAVV(img)
                    zfocus_down_all = np.append(zfocus_down_all, zData_down)
                    fv_down_all = np.append(fv_down_all, fv_midd_down)
                    C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 0 TIMEOUT 1000"})  # AutoFocus Off!
                    C.sendandrecv({"CCP": "WDI SET 4 %f TIMEOUT 1000" % (zData_down - move_z)})
                    time.sleep(1)
                    f.write('tid：%d from down: cyc:%d \tzfocus:%.3f \tfv_value:%.3f \n' % (tid,i, zData_down, fv_midd_down))
            uniformity = max(fv_down_all) - min(fv_down_all)
            with open(file, 'a') as f:
                f.write('tid：%d from up:  \tuniformity:%.3f \n' % (tid, uniformity))
            C.sendandrecv({"CCP": "LED_G CLOSE"})
            f = open(file, 'a')
            localtime = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
            f.write('%s \tFinish!\tAt:%s\n' % (name, localtime))
            f.close()
    def cal_astigmatism(self):
        list_fvB_ML = []
        zData = C.sendandrecv({"CCP": "WDI GET 4 TIMEOUT 1000"})
        # print('wdifocuse=%fum' % zData[b'data'][0])
        z = format(zData[b'data'][0], '.3f')
        zfocus = float(z)
        blueLEDcurrent = 12
        C.sendandrecv({"msgID": 1, "CCP": "LED_B SET 1 %.1f" % blueLEDcurrent})
        C.sendandrecv({"CCP": "CAM SET 3 0 0 2048 2048"})
        zRange = 8
        zStep = 0.2  # um
        localtime0 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        name = localtime0
        outputfolder = 'output\z_get_imge'
        localfolder = com.setOutputFolder(os.path.join('output\get_resolution_image' + '/' + name))



        imgpath = localfolder
        if not os.path.exists(imgpath):
            os.mkdir(imgpath)
        for z in np.arange(zfocus - zRange, zfocus + zRange + zStep, zStep):
            C.sendandrecv({"CCP": "WDI SET 4 %f TIMEOUT 1000" % z})  # 运动到z
            time.sleep(0.3)
            C.sendandrecv({"CCP": "CAM SET 2 0.1"})
            C.sendandrecv({"CCP": "LED_B OPEN"})
            C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 0 0 0"})
            data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
            img = com.data2image(data)
            C.sendandrecv({"CCP": "LED_B CLOSE"})
            imgsmall = img[900:1100, 900:1100]
            #               fvB_WT = com.WAVV(imgsmall)  # 反应照片清晰度的对焦值，小波变换
            fvB_ML = calculation_module.SML(imgsmall)  # 反应照片清晰度的对焦值，改进的拉普拉斯算子
            list_fvB_ML= np.append(list_fvB_ML, fvB_ML)
            cv2.imwrite(os.path.join(imgpath, 'FFimg_z%.2fum_fv%.3f.tiff' % (
                 z, fvB_ML)), img)
            time.sleep(0.02)

    #        max_ind=list_fvB_ML.index(max(list_fvB_ML))
    #        getz=zfocus - zRange+max_ind*0.2+1

    def get_resolution(self,file_pathname):
        Modulation_x_list = np.zeros((200, 9))
        max_value_x = np.zeros((1, 9))
        Modulation_y_list = np.zeros((200, 9))
        max_value_y = np.zeros((1, 9))
        j = 0

        for filename in os.listdir(file_pathname):
            #print(filename)
            full_name = str(file_pathname + '\\' + filename)
            image = (cv2.imread(full_name, cv2.IMREAD_UNCHANGED))
            image_rio = np.empty(shape=(200, 200, 9))
            image_mid = (image[924:1124, 924:1124])
            image_midup = (image[10:210, 924:1124])
            image_middown = (image[1838:2038, 924:1124])
            image_midleft = (image[924:1124, 10:210])
            image_midright = (image[924:1124, 1838:2038])
            image_upright = (image[10:210, 1838:2038])
            image_upleft = (image[10:210, 10:210])
            image_downleft = (image[1838:2038, 10:210])
            image_downright = (image[1838:2038, 1838:2038])
            image_rio[:, :, 0] = (image_mid)
            image_rio[:, :, 1] = (image_midup)
            image_rio[:, :, 2] = (image_middown)
            image_rio[:, :, 3] = (image_midleft)
            image_rio[:, :, 4] = (image_midright)
            image_rio[:, :, 5] = (image_upright)
            image_rio[:, :, 6] = (image_upleft)
            image_rio[:, :, 7] = (image_downleft)
            image_rio[:, :, 8] = (image_downright)

            for i in range(9):
                center, image_8bit =calculation_module.cal_center(image_rio[:, :, i])
                Modulation_x, Modulation_y =  calculation_module.cal_Modulation(center, image_8bit)
                if (np.isnan(Modulation_x)):
                    Modulation_x = 0
                if (np.isnan(Modulation_y)):
                    Modulation_y = 0
                Modulation_x_list[j, i] = Modulation_x
                Modulation_y_list[j, i] = Modulation_y
            j = j + 1


        file = os.path.join(file_pathname +'\\'+ 'resolution.txt')
        f = open(file, 'a')
        for i in range(9):
            max_value_x[0, i] = max(Modulation_x_list[:, i])
            max_value_y[0, i] = max(Modulation_y_list[:, i])
            f.write('cross:%.4f   column:%.4f \n' % (max_value_x[0, i], max_value_y[0, i] ))
        f.close()


    def get_resolution_image(self):

        zData = C.sendandrecv({"CCP": "WDI GET 4 TIMEOUT 1000"})
        # print('wdifocuse=%fum' % zData[b'data'][0])
        z = format(zData[b'data'][0], '.3f')
        zfocus=float(z)
        blueLEDcurrent = 3
        C.sendandrecv({"msgID": 1, "CCP": "LED_B SET 1 %.1f" % blueLEDcurrent})
        C.sendandrecv({"CCP": "CAM SET 3 0 0 2048 2048"})
        zRange = 5
        zStep = 0.2  # um
        localtime0 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        name=localtime0
        outputfolder='output\z_get_imge'
        localfolder = com.setOutputFolder(os.path.join('output\get_resolution_image' + '/' + name))


        for cycle in range(1):

            imgpath = localfolder
            if not os.path.exists(imgpath):
                os.mkdir(imgpath)
            for z in np.arange(zfocus - zRange, zfocus + zRange + zStep, zStep):
                C.sendandrecv({"CCP": "WDI SET 4 %f TIMEOUT 1000" % z})  # 运动到z
                time.sleep(0.3)
                C.sendandrecv({"CCP": "CAM SET 2 0.03"})
                C.sendandrecv({"CCP": "LED_B OPEN"})
                C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 0 0 0"})
                data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
                img = com.data2image(data)
                C.sendandrecv({"CCP": "LED_B CLOSE"})
                imgsmall = img[900:1100, 900:1100]
    #               fvB_WT = com.WAVV(imgsmall)  # 反应照片清晰度的对焦值，小波变换
                fvB_ML = calculation_module.SML(imgsmall)  # 反应照片清晰度的对焦值，改进的拉普拉斯算子
                cv2.imwrite(os.path.join(imgpath, 'FFimg_cycle_%02d_z%.2fum_fv%.3f.tiff' % (
                cycle, z, fvB_ML)), img)
                time.sleep(0.02)
            self.get_resolution(localfolder)

    def Weighted(self, a):
        w = 0
        for i in range(len(a)):
            w = a[i] * (i + 1) + w
        center = w / np.sum(a)
        return center

    def measure_rotation(self):
        C.sendandrecv({"msgID": 1, "CCP": "LED_G SET 1 %.3f" % 0.1})
        C.sendandrecv({"CCP": "LED_G OPEN"})
        localtime0 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        name = localtime0
        outputfolder = 'output\cal_rotation'
        localfolder = com.setOutputFolder(os.path.join('output\scal_rotation' + '/' + name))
        output_img = os.path.join(localfolder, 'cal_rotation.tiff' )
        expTime = float(self.exposure_time.text())
        image = basic_control.cap_image(expTime, 1,output_img)
        C.sendandrecv({"CCP": "LED_G CLOSE"})
        #image = image[200:1800, 1024 - 700:1024 + 700]
        image = image[100:1900, 1200 - 500:1200 + 500]
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        min_16bit = np.min(blur)
        max_16bit = np.max(blur)
        image_8bit = np.array(np.rint(255 * ((blur - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
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
                get_center[i] = self.Weighted(a)
                # get_center[i] = num_peak[0][j] - 10 + center
            # max_center=np.max(get_center)
            # get_center_j = get_center/max_center+get_center_j
            up_coordinate = np.mean(get_center[0:50])
            down_coordinate = np.mean(get_center[-50:-1])
            delta = up_coordinate - down_coordinate
            mean_delta = mean_delta + delta
            # print(get_center)
        mean_delta = mean_delta / len(num_peak)

        cc=bost_UI.Ui_MainWindow.setupUi()
        cc.textBrowser.setText('偏离%.4f像素  ' % (mean_delta))
        self.textBrowser.setText('偏离%.4f像素  ' % (mean_delta))
        file = os.path.join(localfolder, name + '.txt')
        f = open(file, 'a')
        f.write('偏离像素： %.3f \n' %(mean_delta))
        f.close()