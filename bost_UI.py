
#光学测试软件  v0.1.2"

import os
import sys
import cv2
import numpy
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, QTimer, pyqtSignal, QRectF, QSizeF, QPointF
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QFormLayout, QApplication, QWidget
from PyQt5.QtCore import QPointF, Qt, QRectF, QSizeF
from PyQt5.QtGui import QPainter, QColor, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsPixmapItem, QGraphicsScene
from matplotlib import cm, pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import signal
from scipy.signal import argrelextrema
from mpl_toolkits.mplot3d import Axes3D

import Zscan_XY
import common2 as com
import DDS2 as DDS
import numpy as np
import time
import calculation_module
import Background_light_analysis as bla
global C





class z_value_read(QThread): # 创建子线程类
    sinOut_z=pyqtSignal(str) # 自定义信号
   # sinOut_x= pyqtSignal(str)  # 自定义信号
   # sinOut_y = pyqtSignal(str)  # 自定义信号
    def __init__(self):
        super(z_value_read,self).__init__()
    # 重写run()方法
    def run(self):


        while True:

            QThread.msleep(100)
            #T_value = random.randint(200, 225)
            zData = C.sendandrecv({"CCP": "WDI GET 4 TIMEOUT 1000"})
            #print('wdifocuse=%fum' % zData[b'data'][0])
            z=format(zData[b'data'][0], '.3f')
            self.sinOut_z.emit("\n " + str(z) )
           # xData = C.sendandrecv({"CCP": "SERVO01 GET 9 TIMEOUT 5000"})
           # x=format(xData[b'data'][0], '.3f')
           # self.sinOut_x.emit("\n " + str(x) )
           # yData = C.sendandrecv({"CCP": "SERVO02 GET 9 TIMEOUT 5000"})
           # y=format(yData[b'data'][0], '.3f')
           # self.sinOut_y.emit("\n " + str(y))



class WorkThread(QThread):

    save_image_Signal = pyqtSignal()

    def __int__(self):
        super(WorkThread, self).__init__()

    def run(self):
        while True:
            QThread.msleep(1)

class WorkThread2(QThread):

    save_image_Signal = pyqtSignal()

    def __int__(self):
        super(WorkThread2, self).__init__()

    def run(self):
        while True:
            QThread.msleep(1)

        #self.trigger.emit()


class Ui_MainWindow(object):
    global C
    def setupUi(self, MainWindow):
        global C
        global Xoffset, Yoffset

        C, self.outputfolder, self.logger,Xoffset, Yoffset=self.init_ds()
        MainWindow.setObjectName("光学测试软件  v0.1.3")
        MainWindow.resize(1300, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.chip_in_button = QtWidgets.QPushButton(self.centralwidget)# 进仓触发
        self.chip_in_button.setGeometry(QtCore.QRect(80, 60, 93, 28))
        self.chip_in_button.setObjectName("chip_in_button")

        self.chip_out_button = QtWidgets.QPushButton(self.centralwidget)#出仓触发
        self.chip_out_button.setGeometry(QtCore.QRect(80, 110, 93, 28))
        self.chip_out_button.setObjectName("chip_out_button")

        self.z_move_button = QtWidgets.QPushButton(self.centralwidget)#z轴触发
        self.z_move_button.setGeometry(QtCore.QRect(220, 60, 93, 28))
        self.z_move_button.setObjectName("z_move_button")

        self.x_move_button = QtWidgets.QPushButton(self.centralwidget)#x轴触发
        self.x_move_button.setGeometry(QtCore.QRect(340, 60, 93, 28))
        self.x_move_button.setObjectName("x_move_button")

        self.y_move_button = QtWidgets.QPushButton(self.centralwidget)#y轴触发
        self.y_move_button.setGeometry(QtCore.QRect(450, 60, 93, 28))
        self.y_move_button.setObjectName("y_move_button")

        self.close_button = QtWidgets.QPushButton(self.centralwidget)#关闭触发
        self.close_button.setGeometry(QtCore.QRect(80, 160, 93, 28))
        self.close_button.setObjectName("close_button")

        self.x_reset_button = QtWidgets.QPushButton(self.centralwidget)#x复位触发
        self.x_reset_button.setGeometry(QtCore.QRect(330, 300, 93, 28))
        self.x_reset_button.setObjectName("x_reset_button")

        self.y_reset_button = QtWidgets.QPushButton(self.centralwidget)#y复位触发
        self.y_reset_button.setGeometry(QtCore.QRect(450, 300, 93, 28))
        self.y_reset_button.setObjectName("y_reset_button")

        self.x_value = QtWidgets.QLineEdit(self.centralwidget) #x轴移动量
        self.x_value.setGeometry(QtCore.QRect(340, 110, 91, 21))
        self.x_value.setObjectName("x_value")
        self.x_value.setText(str(0))

        self.y_value = QtWidgets.QLineEdit(self.centralwidget)#y轴移动量
        self.y_value.setGeometry(QtCore.QRect(450, 110, 91, 21))
        self.y_value.setObjectName("y_value")
        self.y_value.setText(str(0))

        self.fouce_on_button = QtWidgets.QPushButton(self.centralwidget) #开启对焦触发
        self.fouce_on_button.setGeometry(QtCore.QRect(70, 300, 93, 28))
        self.fouce_on_button.setObjectName("fouce_on_button")

        self.fouce_off_buttion = QtWidgets.QPushButton(self.centralwidget)#关闭对焦触发
        self.fouce_off_buttion.setGeometry(QtCore.QRect(70, 340, 93, 28))
        self.fouce_off_buttion.setObjectName("fouce_off_buttion")

        self.make_0_button = QtWidgets.QPushButton(self.centralwidget) #make0对焦触发
        self.make_0_button.setGeometry(QtCore.QRect(210, 300, 93, 28))
        self.make_0_button.setObjectName("make_0_button")

        self.z_value = QtWidgets.QLineEdit(self.centralwidget) #Z轴移动量
        self.z_value.setGeometry(QtCore.QRect(220, 110, 91, 21))
        self.z_value.setObjectName("z_value")
        self.z_value.setText(str(0))

        self.fluorsecent_value = QtWidgets.QLineEdit(self.centralwidget) #荧光电流值
        self.fluorsecent_value.setGeometry(QtCore.QRect(70, 430, 91, 21))
        self.fluorsecent_value.setObjectName("fluorsecent_value")
        self.fluorsecent_value.setText(str(10))

        self.Brightfield_value = QtWidgets.QLineEdit(self.centralwidget)#明场电流值
        self.Brightfield_value.setGeometry(QtCore.QRect(200, 430, 91, 21))
        self.Brightfield_value.setObjectName("Brightfield_value")
        self.Brightfield_value.setText(str(0.2))

        self.y_positive_button = QtWidgets.QPushButton(self.centralwidget)#Y轴正向移动触发
        self.y_positive_button.setGeometry(QtCore.QRect(450, 150, 93, 28))
        self.y_positive_button.setObjectName("y_positive_button")

        self.y_negative_button = QtWidgets.QPushButton(self.centralwidget)#Y轴负向移动触发
        self.y_negative_button.setGeometry(QtCore.QRect(450, 230, 93, 28))
        self.y_negative_button.setObjectName("y_negative_button")

        self.x_negative_button = QtWidgets.QPushButton(self.centralwidget)#X轴负向移动触发
        self.x_negative_button.setGeometry(QtCore.QRect(340, 230, 93, 28))
        self.x_negative_button.setObjectName("x_negative_button")

        self.x_positive_button = QtWidgets.QPushButton(self.centralwidget)#X轴正向移动触发
        self.x_positive_button.setGeometry(QtCore.QRect(340, 150, 93, 28))
        self.x_positive_button.setObjectName("x_positive_button")

        self.z_up_button = QtWidgets.QPushButton(self.centralwidget)#z轴正向移动触发
        self.z_up_button.setGeometry(QtCore.QRect(220, 150, 93, 28))
        self.z_up_button.setObjectName("z_up_button")
        self.z_up_button.setText(str(0))

        self.z_down_button = QtWidgets.QPushButton(self.centralwidget)#z轴负向移动触发
        self.z_down_button.setGeometry(QtCore.QRect(220, 230, 93, 28))
        self.z_down_button.setObjectName("z_down_button")
        self.z_down_button.setText(str(0))

        self.z_setp_value = QtWidgets.QLineEdit(self.centralwidget)  # z移动步长
        self.z_setp_value.setGeometry(QtCore.QRect(220, 190, 91, 21))
        self.z_setp_value.setObjectName("z_setp_value")
        self.z_setp_value.setText(str(0.5))

        self.y_setp_value = QtWidgets.QLineEdit(self.centralwidget)  # y移动步长
        self.y_setp_value.setGeometry(QtCore.QRect(450, 190, 91, 21))
        self.y_setp_value.setObjectName("y_setp_value")
        self.y_setp_value.setText(str(0.1))

        self.x_setp_value = QtWidgets.QLineEdit(self.centralwidget)  # x移动步长
        self.x_setp_value.setGeometry(QtCore.QRect(340, 190, 91, 21))
        self.x_setp_value.setObjectName("x_setp_value")
        self.x_setp_value.setText(str(0.1))

        self.z_display = QtWidgets.QLineEdit(self.centralwidget)  # 显示当前Z值
        self.z_display.setGeometry(QtCore.QRect(220, 30, 91, 21))
        self.z_display.setDragEnabled(False)
        self.z_display.setObjectName("z_display")
        self.z_display.setFocusPolicy(QtCore.Qt.NoFocus)

        self.x_display = QtWidgets.QLineEdit(self.centralwidget)  # 显示当前X值
        self.x_display.setGeometry(QtCore.QRect(340, 30, 91, 21))
        self.x_display.setFocusPolicy(QtCore.Qt.NoFocus)
        self.x_display.setDragEnabled(False)
        self.x_display.setObjectName("x_display")

        self.y_display = QtWidgets.QLineEdit(self.centralwidget)  # 显示当前Y值
        self.y_display.setGeometry(QtCore.QRect(450, 30, 91, 21))
        self.y_display.setDragEnabled(False)
        self.y_display.setObjectName("y_display")
        self.y_display.setFocusPolicy(QtCore.Qt.NoFocus)

        self.fluorsecent_on_button = QtWidgets.QPushButton(self.centralwidget)#荧光开启触发
        self.fluorsecent_on_button.setGeometry(QtCore.QRect(70, 480, 93, 28))
        self.fluorsecent_on_button.setObjectName("fluorsecent_on_button")

        self.Brightfield_on_button = QtWidgets.QPushButton(self.centralwidget)#明场开启触发
        self.Brightfield_on_button.setGeometry(QtCore.QRect(200, 480, 93, 28))
        self.Brightfield_on_button.setObjectName("Brightfield_on_button")

        self.fluorsecent_off_button = QtWidgets.QPushButton(self.centralwidget)#荧光关闭触发
        self.fluorsecent_off_button.setGeometry(QtCore.QRect(70, 520, 93, 28))
        self.fluorsecent_off_button.setObjectName("fluorsecent_off_button")

        self.Brightfield_off_button = QtWidgets.QPushButton(self.centralwidget)#明场关闭触发
        self.Brightfield_off_button.setGeometry(QtCore.QRect(200, 520, 93, 28))
        self.Brightfield_off_button.setObjectName("Brightfield_off_button")

        self.label_flur = QtWidgets.QLabel(self.centralwidget)#label 显示
        self.label_flur.setGeometry(QtCore.QRect(90, 400, 72, 20))
        self.label_flur.setObjectName("label_flur")
        self.label_Bright = QtWidgets.QLabel(self.centralwidget)
        self.label_Bright.setGeometry(QtCore.QRect(220, 400, 72, 20))
        self.label_Bright.setObjectName("label_Bright")
        self.label_wdi = QtWidgets.QLabel(self.centralwidget)
        self.label_wdi.setGeometry(QtCore.QRect(20, 300, 72, 20))
        self.label_wdi.setObjectName("label_wdi")
        self.label_wdi_up = QtWidgets.QLabel(self.centralwidget)
        self.label_wdi_up.setGeometry(QtCore.QRect(410, 350, 72, 20))
        self.label_wdi_up.setObjectName("label_wdi_up")
        self.label_wdi_down = QtWidgets.QLabel(self.centralwidget)
        self.label_wdi_down.setGeometry(QtCore.QRect(410, 380, 72, 20))
        self.label_wdi_down.setObjectName("label_wdi_down")
        self.label_wdi_step = QtWidgets.QLabel(self.centralwidget)
        self.label_wdi_step.setGeometry(QtCore.QRect(410, 410, 72, 20))
        self.label_wdi_step.setObjectName("label_wdi_step")

        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)#显示图像
        self.graphicsView.setGeometry(QtCore.QRect(610, 20, 630, 630))
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.verticalScrollBar().setSingleStep(0)
        self.graphicsView.horizontalScrollBar().setSingleStep(0)
        self.graphicsView.setObjectName("graphicsView")

        self.bright_capture_button = QtWidgets.QPushButton(self.centralwidget)#单帧拍图存储
        self.bright_capture_button.setGeometry(QtCore.QRect(60, 570, 101, 31))
        self.bright_capture_button.setObjectName("bright_capture_button")

        self.zscan_button = QtWidgets.QPushButton(self.centralwidget)#明场z扫描对焦
        self.zscan_button.setGeometry(QtCore.QRect(210, 340, 93, 28))
        self.zscan_button.setObjectName("zscan_button")

        self.scan_up_limit = QtWidgets.QLineEdit(self.centralwidget)  #z扫描上限
        self.scan_up_limit.setGeometry(QtCore.QRect(450, 350, 91, 21))
        self.scan_up_limit.setObjectName("scan_up_limit")
        self.scan_up_limit.setText(str(360))

        self.scan_down_limit = QtWidgets.QLineEdit(self.centralwidget)#z扫描下限
        self.scan_down_limit.setGeometry(QtCore.QRect(450, 380, 91, 21))
        self.scan_down_limit.setObjectName("scan_down_limit")
        self.scan_down_limit.setText(str(350))


        self.scan_step_value = QtWidgets.QLineEdit(self.centralwidget)#z扫描步长
        self.scan_step_value.setGeometry(QtCore.QRect(450, 410, 91, 21))
        self.scan_step_value.setObjectName("scan_step_value")
        self.scan_step_value.setText(str(0.5))

        self.zscan_fcouse_value = QtWidgets.QLineEdit(self.centralwidget)#Z扫描对焦值
        self.zscan_fcouse_value.setGeometry(QtCore.QRect(330, 350, 61, 21))
        self.zscan_fcouse_value.setObjectName("zscan_fcouse_value")

        self.continuous_capture_button_on = QtWidgets.QPushButton(self.centralwidget)#连续采集开启
        self.continuous_capture_button_on.setGeometry(QtCore.QRect(60, 620, 101, 31))
        self.continuous_capture_button_on.setObjectName("continuous_capture_button_on")

        self.continuous_capture_button_off = QtWidgets.QPushButton(self.centralwidget)#连续采集关闭
        self.continuous_capture_button_off.setGeometry(QtCore.QRect(200, 620, 101, 31))
        self.continuous_capture_button_off.setObjectName("continuous_capture_button_off")


        self.exposure_time = QtWidgets.QLineEdit(self.centralwidget) #曝光时间
        self.exposure_time.setGeometry(QtCore.QRect(310, 430, 80, 25))
        self.exposure_time.setObjectName("exposure_time")
        self.exposure_time_label = QtWidgets.QLabel(self.centralwidget)
        self.exposure_time_label.setGeometry(QtCore.QRect(310, 390, 81, 31))
        self.exposure_time_label.setObjectName("exposure_time_label")

        self.shading_button = QtWidgets.QPushButton(self.centralwidget)#照明测量触发
        self.shading_button.setGeometry(QtCore.QRect(450, 570, 121, 31))
        self.shading_button.setObjectName("shading_button")

        self.Resolution_button = QtWidgets.QPushButton(self.centralwidget)#分辨率测试触发
        self.Resolution_button.setGeometry(QtCore.QRect(450, 617, 121, 31))
        self.Resolution_button.setObjectName("Resolution_button")

        self.Nine_View_button = QtWidgets.QPushButton(self.centralwidget)#显示9宫格触发
        self.Nine_View_button.setGeometry(QtCore.QRect(200, 570, 101, 31))
        self.Nine_View_button.setObjectName("Resolution_button")

        self.cal_rotation_button = QtWidgets.QPushButton(self.centralwidget)#相机旋转测试
        self.cal_rotation_button.setGeometry(QtCore.QRect(310, 525, 121, 31))
        self.cal_rotation_button.setObjectName("Resolution_button")

        self.WDI_test_button = QtWidgets.QPushButton(self.centralwidget) #WDI对焦精度测试
        self.WDI_test_button.setGeometry(QtCore.QRect(310, 570, 121, 31))
        self.WDI_test_button.setObjectName("WDI_test_button")

        self.static_focus_button = QtWidgets.QPushButton(self.centralwidget)#WDI重复对焦精度测试
        self.static_focus_button.setGeometry(QtCore.QRect(310, 620, 121, 31))
        self.static_focus_button.setObjectName("static_focus_button")

        self.Background_light_test_button = QtWidgets.QPushButton(self.centralwidget) #背景值对焦精度测试
        self.Background_light_test_button.setGeometry(QtCore.QRect(310, 480, 130, 31))
        self.Background_light_test_button.setObjectName("Background_light_test_button")

        self.flat_button = QtWidgets.QPushButton(self.centralwidget)#调平测试
        self.flat_button.setGeometry(QtCore.QRect(450, 480, 101, 31))
        self.flat_button.setObjectName("flat_button")

        self.astigmatism_button = QtWidgets.QPushButton(self.centralwidget)#像散测试
        self.astigmatism_button.setGeometry(QtCore.QRect(450, 530, 101, 31))
        self.astigmatism_button.setObjectName("astigmatism_button")
        self.exposure_time.setText(str(0.003))

        self.Z_MAP_button = QtWidgets.QPushButton(self.centralwidget)#Z-FFT
        self.Z_MAP_button.setGeometry(QtCore.QRect(480, 670, 101, 31))
        self.Z_MAP_button.setObjectName("Z_MAP_button")

        self.Field_curvature_button = QtWidgets.QPushButton(self.centralwidget)
        self.Field_curvature_button.setGeometry(QtCore.QRect(480, 700, 101, 31))
        self.Field_curvature_button.setObjectName("Field_curvature_button")

        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)#内容显示
        self.textBrowser.setGeometry(QtCore.QRect(60, 670, 400, 100))
        self.textBrowser.setObjectName("textBrowser")

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1545, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)


        self.retranslateUi(MainWindow)  #触发进程
        self.close_button.clicked.connect(self.exit_all)
        self.chip_in_button.clicked.connect(self.chip_in)
        self.chip_out_button.clicked.connect(self.chip_out)
        self.x_reset_button.clicked.connect(self.x_move_resets)
        self.y_reset_button.clicked.connect(self.y_move_resets)
        self.Brightfield_on_button.clicked.connect(self.Brightfield_on_act)
        self.Brightfield_off_button.clicked.connect(self.Brightfield_off_act)
        self.x_move_button.clicked.connect(lambda :self.move_x(self.x_value.text()))
        self.y_move_button.clicked.connect(lambda :self.move_y(self.y_value.text()))
        self.z_move_button.clicked.connect(lambda :self.move_z(self.z_value.text()))
        self.bright_capture_button.clicked.connect(self.sig_image_act)
        self.fouce_off_buttion.clicked.connect( self.fouce_off)
        self.fouce_on_button.clicked.connect(self.fouce_on)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.z_read = z_value_read()  # 创建子线程对象
        self.timer = QTimer()
        self.timer2 = QTimer()
        self.workThread = WorkThread()
        self.workThread2 = WorkThread2()
        self.continuous_capture_button_on.clicked.connect(self.continuous_capture_process)
        self.timer.timeout.connect(self.start_camer)
        self.timer2.timeout.connect(self.start_nine)
        self.Nine_View_button.clicked.connect(self.Nine_View_capture_process)
        self.continuous_capture_button_off.clicked.connect(self.stop_camer)
        self.z_read.sinOut_z.connect(self.diplay_z_value)  # 将线程信号连接到槽函数
        #self.z_read.sinOut_x.connect(self.get_x_display)  # 将线程信号连接到槽函数
        #self.z_read.sinOut_y.connect(self.get_y_display)  # 将线程信号连接到槽函数
        self.x_positive_button.clicked.connect(lambda :self.move_x_step(1))
        self.y_positive_button.clicked.connect(lambda :self.move_y_step(1))
        self.z_up_button.clicked.connect(lambda :self.move_z_step(1))
        self.x_negative_button.clicked.connect( lambda :self.move_x_step(0))
        self.y_negative_button.clicked.connect( lambda :self.move_y_step(0))
        self.z_down_button.clicked.connect( lambda :self.move_z_step(0))
        self.make_0_button.clicked.connect(self.make_0_start)
        self.fluorsecent_on_button.clicked.connect(self.fluorsecent_on)
        self.fluorsecent_off_button.clicked.connect(self.fluorsecent_off)
        self.shading_button.clicked.connect(self.measure_shading)
        self.Background_light_test_button.clicked.connect(self.Background_light_test)
        self.Resolution_button.clicked.connect(self.measure_resolution)
        self.cal_rotation_button.clicked.connect(self.measure_rotation)
        self.static_focus_button.clicked.connect(self.static_focus)
        self.WDI_test_button.clicked.connect(self.WDI_test)
        self.zscan_button.clicked.connect(self.zscan_start)
        self.astigmatism_button.clicked.connect(self.means_astigmatism)
        self.Z_MAP_button.clicked.connect(self.measure_z_map)
        self.Field_curvature_button.clicked.connect(self.Field_curvature)

        self.get_x_display()
        self.get_y_display()
        self.get_z_display()



    def zscan_start(self): # z扫描对焦并移动到最佳对焦位置

        C.sendandrecv({"msgID": 1, "CCP": "LED_G SET 1 %.1f" % 0.2})
        C.sendandrecv({"CCP": "CAM SET 2 %.3f" % 0.002})
        self.get_x_display()
        self.get_y_display()
        C.sendandrecv({"CCP": "CAM SET 3 0 0 2048 2048"})
        C.sendandrecv({"CCP": "LED_G OPEN"})
        up=float(self.scan_up_limit.text())
        down=float(self.scan_down_limit.text())
        step=float(self.scan_step_value.text())
        z, fv, imgs, zFocus = self.zscan_get_focus(down, up, step, show=True)
        C.sendandrecv({"CCP": "LED_G CLOSE"})
        C.sendandrecv({"CCP": "WDI SET 4 %f TIMEOUT 1000" % zFocus})
        self.zscan_fcouse_value.setText(str(zFocus))
        self.get_z_display()


    def zscan_get_focus(self,start, end, step, show=False, **kwargs):#z扫描对焦获得最佳位置
        images = []
        fv = np.array([])
        zs = np.arange(start, end, step)
        C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 0 TIMEOUT 1000"})  # WDI
        C.sendandrecv({"msgID": 1, "CCP": "WDI SET 4 %f TIMEOUT 1000" % zs[0]})  # WDI
        time.sleep(0.1)

        for z in zs:
            C.sendandrecv({"CCP": "WDI SET 4 %f TIMEOUT 1000" % z}) # WDI
            C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 1 0 0"})  # WDI
            data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
            img = com.data2image(data)
            images.append(img)
            imgsmall = img[900:1100, 900:1100]
            fv = np.append(fv, calculation_module.WAVV(imgsmall))
        if show:
            plt.plot(zs, fv)
            plt.show()
        zfouse=zs[fv.argmax()]
        zfouse=round(zfouse, 3)
        return zs, fv, images,zfouse


    def show_image(self, image, scale=0.3): #显示图像
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
        #self.setSceneDims()
        self.graphicsView.setRenderHints(QPainter.Antialiasing | QPainter.HighQualityAntialiasing |
                                         QPainter.SmoothPixmapTransform)
        self.graphicsView.setViewportUpdateMode(self.graphicsView.SmartViewportUpdate)
        self.scene = QGraphicsScene()  # 创建场景s
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)

    # def start(self):
    #     self.z_read.start()

    def continuous_capture_process(self): #连续采集进程
        # 计时器每秒计数
        self.timer2.stop()
        self.timer.start(30)
        self.workThread.start()


    def Nine_View_capture_process(self):#9宫格连续采集进程
        # 计时器每秒计数
        self.stop_camer
        self.timer2.start(30)
        # 计时开始
        self.workThread2.start()

    def start_camer(self):
        self.save_image_act()

    def start_nine(self):
        self.timer.stop()
        self.nine_View()

    def stop_camer(self):
        self.timer.stop()
        self.timer2.stop()

    def get_z_display(self): #获取z值
        zData = C.sendandrecv({"CCP": "WDI GET 4 TIMEOUT 1000"})
        # print('wdifocuse=%fum' % zData[b'data'][0])
        z = format(zData[b'data'][0], '.3f')
        self.z_display.setText(z)

    def diplay_z_value(self, str):#显示z位置
        self.z_display.setText(str)

    def get_x_display(self):#显示x位置
        xData = C.sendandrecv({"CCP": "SERVO01 GET 9 TIMEOUT 5000"})
        x = format(xData[b'data'][0], '.3f')
        self.x_display.setText(x)

    def get_y_display(self):#显示y位置
        yData = C.sendandrecv({"CCP": "SERVO02 GET 9 TIMEOUT 5000"})
        y = format(yData[b'data'][0], '.3f')
        self.y_display.setText(y)

    def init_ds(self):#初始化ds
        outputfolder = com.setOutputFolder('output')
        logger = com.createLogger(outputfolder)
        com.logger = logger
        com.outputfolder = outputfolder
        com.tilemap = com.TileMap('save/TM518.txt')
        readTMdefine = com.readTileMap
        C = DDS.NSDS(outputfolder)
        # ip = '10.0.32.101'
        ip = '127.0.0.1'
        C.init_msgclient(ip, '6666')
        C.init_datreceiver(ip, '7777')
        C.init_imgreceiver(ip, '20183')
        com.C = C
        #C._init_LED()
        C.MSG = {}
        dct_tm, dct_ij, dct_param = readTMdefine('save/TM518.txt')
        Xoffset = dct_param['XOffset']
        Yoffset = dct_param['YOffset']
        return C, outputfolder, logger, Xoffset, Yoffset

    def chip_in(self):
        # Chip In
        C.sendandrecv({"CCP": "MOTOR_C01 MOV 2"})
        time.sleep(10)
        C.sendandrecv({"CCP": "MOTOR_C02 MOV 2"})

        self.get_y_display()
        self.get_x_display()
        self.textBrowser.clear()
        self.textBrowser.setText('进仓结束')

    def chip_out(self):
        # Chip out
        C.sendandrecv({"CCP": "SERVO01 RESET"})
        C.sendandrecv({"CCP": "SERVO02 RESET"})
        time.sleep(10)
        C.sendandrecv({"CCP": "MOTOR_C02 MOV 1"})
        time.sleep(10)
        C.sendandrecv({"CCP": "MOTOR_C01 MOV 1"})

        self.textBrowser.clear()
        self.textBrowser.setText('出仓结束')

    def x_move_resets(self):
        C.sendandrecv({"CCP": "SERVO01 RESET"})
        self.get_x_display()

    def y_move_resets(self):
        C.sendandrecv({"CCP": "SERVO02 RESET"})
        self.get_y_display()
#按照tile 移动
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
        self.textBrowser.clear()
        self.textBrowser.setText('明场灯开')
    def Brightfield_off_act(self):
        C.sendandrecv({"CCP": "LED_G CLOSE"})
        self.textBrowser.clear()
        self.textBrowser.setText('明场灯关')
    def fluorsecent_on(self):
        blueLEDcurrent = float(self.fluorsecent_value.text())
        C.sendandrecv({"msgID": 1, "CCP": "LED_B SET 1 %.3f" % blueLEDcurrent})
        C.sendandrecv({"CCP": "LED_B OPEN"})
        self.textBrowser.clear()
        self.textBrowser.setText('荧光场灯开')
    def fluorsecent_off(self):
        C.sendandrecv({"CCP": "LED_B CLOSE"})
        self.textBrowser.clear()
        self.textBrowser.setText('荧光场灯关')

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

    def save_image_act(self):  #存储当前图像
        expTime = float(self.exposure_time.text())
        image = self.cap_image(expTime, 0)
        image = calculation_module.image_16_8(image)
        self.show_image(image)
        # outputfolder = self.com.setOutputFolder('output\BOST_FOR_FCOUSE')
        # cv2.imwrite(outputfolder + r'\test.tiff', image)

    def fouce_off(self):
        C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 0 TIMEOUT 1000"})
        self.z_read.terminate()
        self.textBrowser.clear()
        self.textBrowser.setText('自动对焦关闭')
    def fouce_on(self):
        C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 1 TIMEOUT 1000"})

        self.z_read.start()
        self.textBrowser.clear()
        self.textBrowser.setText('自动对焦开启')

    def make_0_start(self):
        C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 0 TIMEOUT 1000"})
        print('AutoFocus Off!')
        C.sendandrecv({"CCP": "WDI DEVICECONTROL 3 TIMEOUT 5000"})
        print('Make0 OK!')
        self.textBrowser.clear()
        self.textBrowser.setText('make0 完毕')
    def setBackground(self, color):

        if isinstance(color, QColor):
            self.graphicsView.setBackgroundBrush(color)
        elif isinstance(color, (str, Qt.GlobalColor)):
            color = QColor(color)
            if color.isValid():
                self.graphicsView.setBackgroundBrush(color)


    def cap_image(self, expTime, save, outputfolder=None):

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
        if save == 1:
            cv2.imwrite(outputfolder, img)
        return img

    def close_all(self):
        C.sendandrecv({"CCP": "LED_B CLOSE"})
        C.sendandrecv({"CCP": "LED_G CLOSE"})
        C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 0 TIMEOUT 1000"})
        self.stop_camer()
    def exit_all(self):
        self.close_all()
        sys.exit(0)

    def measure_shading(self):
        self.close_all()
        blueLEDcurrent = float(self.fluorsecent_value.text())
        C.sendandrecv({"msgID": 1, "CCP": "LED_B SET 1 %.1f" % blueLEDcurrent})
        C.sendandrecv({"CCP": "LED_B OPEN"})
        localtime0 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        name = localtime0
        localfolder = com.setOutputFolder(os.path.join('output\shading' + '/' + name))
        output_shading_plot = os.path.join(localfolder, 'shading_plot.png')
        output_img = os.path.join(localfolder, 'shading.tiff')
        expTime = float(self.exposure_time.text())
        img = self.cap_image(expTime, 1, output_img)
        C.sendandrecv({"CCP": "LED_B CLOSE"})
        Uniformity, max_value, min_value, img_resize_down, img_resize_up,mean_value = self.cal_light_Uniformity(img)
        plt = self.polt_3d(img_resize_down)
        plt.savefig(os.path.join(output_shading_plot))
        img_show = calculation_module.image_16_8(img_resize_up)
        self.show_image(img_show)
        f = open(os.path.join(localfolder, 'shading.txt'), 'a')
        f.write('max_value=%.3f\nmin_value=%.3f\nUniformity=%.3f\nmean_value=%.3f\n' % (max_value, min_value,Uniformity,mean_value))
        f.close()
        self.textBrowser.clear()
        self.textBrowser.setText('照明测试结果\n')
        self.textBrowser.setText('max_value=%.3f\tmin_value=%.3f\tUniformity=%.3f\tnmean_value=%.3f\n' % (max_value, min_value,Uniformity,mean_value))

    def cal_light_Uniformity(self, img):
        self.close_all()
        img_mean = cv2.blur(img, (5, 5))
        img_resize_down = cv2.resize(img_mean, dsize=(100, 100),
                                     interpolation=cv2.INTER_NEAREST)
        img_resize_up = cv2.resize(img_resize_down, dsize=(2048, 2048),
                                   interpolation=cv2.INTER_LINEAR)
        maxindex = np.where(img_resize_up == np.max(img_resize_up))
        minindex = np.where(img_resize_up == np.min(img_resize_up))
        max_value = np.max(img_resize_down)
        min_value = np.min(img_resize_down)
        mean_value = np.mean(img_resize_down)
        Uniformity_max = 1 - (max_value - mean_value) / mean_value
        Uniformity_min = 1 - (mean_value - min_value) / mean_value
        Uniformity = min(Uniformity_max, Uniformity_min) * 100

        return Uniformity, max_value, min_value, img_resize_down, img_resize_up,mean_value

    def polt_3d(self, img):
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
        maxz = np.max(z)
        minz = np.min(z)
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
        #plt.savefig(os.path.join('output\shading\shading_plot.png'))
        return plt


    def Background_light_test(self):
        self.close_all
        x = Xoffset + 17.98 / 2
        y = Yoffset + 48.6 / 2
        C.sendandrecv({"CCP": "SERVO01 MOV 0 %.2f 0" % (x)})
        C.sendandrecv({"CCP": "SERVO02 MOV 0 %.2f 0" % (y)})
        localtime0 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        move_localfolder = com.setOutputFolder('output\Background_light_test\\light\\' + localtime0)
        outputfolder = com.setOutputFolder('output\Background_light_test\\nolight\\' + localtime0)
        output_img = os.path.join(outputfolder, 'no_light.tiff')
        expTime = 0.03
        img = self.cap_image(expTime, 1, output_img)
        img=img[2:2045,2:2045]
        img = cv2.blur(img, (5, 5), 0)
        mean_img = np.mean(img)
        max = np.max(img)
        min = np.min(img)
        f = open(os.path.join(outputfolder, 'no_light.txt'), 'a')
        f.write('mean=%.3f\tmax=%.3f\tmin=%.3f\n' % (mean_img, max,min))
        f.close()
        FF_time=0.03
        FF_LED=10

        self.YX_scan_FF_image(FF_time,FF_LED,move_localfolder)
        self.textBrowser.clear()
        self.textBrowser.setText('背景光测试结束')

        bla.backguround_ligt_result(move_localfolder)

    def YX_scan_FF_image(self,FF_time,FF_LED,move_localfolder):
        # 扫描对焦

        global tilenumber
        cyc = 1
        FMfile = 'save/FM518_S.txt'
        FM, FMZ = com.readFocusMap(FMfile)
        tilenumber = len(FM)
        tilemap = com.TileMap('save/TM518.txt')
        expTime = 0.03
        outputfolder = com.setOutputFolder('output/Background_light_test')
        blueLEDcurrent = 10
        C.sendandrecv({"CCP": "CAM SET 2 %.3f" % FF_time})
        C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 1 TIMEOUT 1000"})  # AutoFocus On!
        C.sendandrecv({"CCP": "SERVO01 MOV 4 %.2f 0"})
        C.sendandrecv({"CCP": "SERVO02 MOV 4 %.2f 0"})
        wait_time = float(0.1)
        time.sleep(wait_time)
        fv_tile = []
        zfocus_wdi_tile = []
        #localtime0 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
       # move_localfolder = com.setOutputFolder('output\Background_light_test\\' + localtime0)
        C.sendandrecv({"msgID": 1, "CCP": "LED_B SET 1 %.1f" % FF_LED})
        # C.sendandrecv({"CCP": "LED_G OPEN"})
        # C.sendandrecv({"CCP": "LED_B OPEN"})
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
               # zData = C.sendandrecv({"CCP": "WDI GET 5 TIMEOUT 1000"})  # To Wait for FocusOver and Get z
               # zData_down = float(zData[b'data'][0])
               # C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 0 TIMEOUT 1000"})  # AutoFocus Off!
               # C.sendandrecv({"CCP": "WDI SET 4 %f TIMEOUT 1000" % (zData_down - 498.5)})
                C.sendandrecv({"CCP": "CAM SET 3 0 0 2048 2048"})
                C.sendandrecv({"CCP": "LED_B OPEN"})
                C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 0 0 0"})
                data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
                C.sendandrecv({"CCP": "LED_B CLOSE"})
                wdi_img = com.data2image(data, [2048, 2048])
                output_img = os.path.join(move_localfolder, 'wdi_tile%04d_%02d.tiff' % (tid, i))
                cv2.imwrite(output_img, wdi_img)
                # C.sendandrecv({"CCP": "WDI SET 4 %f TIMEOUT 1000" % (zData_down )})
        C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 0 TIMEOUT 1000"})  # AutoFocus Off!

    def WDI_test(self):
        self.close_all
        zRange=2.5
        zStep=0.25
        FF_expTime=float(self.exposure_time.text())
        BF_expTime=0.002
        BFLEDcurrent = float(self.Brightfield_value.text())
        FFLEDcurrent = float(self.fluorsecent_value.text())
        FMfile = 'save/WDI_MAP.txt'
        FM, FMZ = com.readFocusMap(FMfile)
        tilenumber = len(FM)
        tilemap = com.TileMap('save/TM518.txt')
        localtime0 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        runID = localtime0
        C.sendandrecv({"CCP": "WDI_CAPTURER FOCUS"})
        C.sendandrecv({"CCP": "CAM SET 3 0 0 2048 2048"})
        for tid in FM:
            xy = tilemap.t2xy([tid])[0]
            x = xy[0]
            y = xy[1]
            C.sendandrecv({"CCP": "SERVO01 MOV 0 %.2f 0" % (x)})
            C.sendandrecv({"CCP": "SERVO02 MOV 0 %.2f 0" % (y)})

            self.z_get_imge(x, y, zRange, zStep, FF_expTime, BF_expTime, BFLEDcurrent, FFLEDcurrent, runID)


        resultfolder = com.setOutputFolder(os.path.join('output\z_get_imge'))
        result_path = resultfolder
        save_path = resultfolder

        Zscan_XY.Zscan_data_extract_singlechip(runID, result_path, save_path)

    def z_get_imge(self,x,y,zRange,zStep,FF_expTime,BF_expTime,BFLEDcurrent,FFLEDcurrent,runID):

       # greenLEDcurrent = float(self.Brightfield_value.text())
       # blueLEDcurrent = float(self.fluorsecent_value.text())

        C.sendandrecv({"msgID": 1, "CCP": "LED_G SET 1 %.1f" % BFLEDcurrent})
        C.sendandrecv({"msgID": 1, "CCP": "LED_B SET 1 %.1f" % FFLEDcurrent})
        C.sendandrecv({"CCP": "CAM SET 3 0 0 2048 2048"})

       # zRange = 8
       # zStep = 0.2  # um

        name = ("X%.2f,Y%.2f"%(x,y))

        outputfolder = 'output\z_get_imge'
        localfolder = com.setOutputFolder(os.path.join('output\z_get_imge'+ '\\'+runID + '\\' + name))
        file = os.path.join(localfolder, name + '.txt')
        f = open(file, 'a')

        f.write('Start test\tzRange=%f\tzStep=%f\n' % (zRange, zStep))
        f.close()
        expTime = float(self.exposure_time.text())

        for cycle in range(1):
            f = open(file, 'a')
            localtime = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
            print('cycle:%d\tAt:%s' % (cycle, localtime))
            f.write('cycle:%d\tAt:%s\n' % (cycle, localtime))
            f.close()

            C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 1 TIMEOUT 1000"})  # AutoFocus On!
            time.sleep(0.5)
            C.sendandrecv({"CCP": "WDI GET 5 TIMEOUT 1000"})
            zData = C.sendandrecv({"CCP": "WDI GET 4 TIMEOUT 1000"})
            # print('wdifocuse=%fum' % zData[b'data'][0])
            z = format(zData[b'data'][0], '.3f')
            zfocus = float(z)
            C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 0 TIMEOUT 1000"})  # AutoFocus Off!

            with open(file, 'a') as f:
                imgpath = localfolder + '/img_cycle%02d' % (cycle)
                if not os.path.exists(imgpath):
                    os.mkdir(imgpath)

                #C.sendandrecv({"CCP": "CAM SET 3 0 0 2048 2048"})
                print('zfocus:', zfocus)
                C.sendandrecv({"CCP": "WDI SET 4 %f TIMEOUT 1000" % zfocus})  # 运动到z
                print('cycle:%02d \tzfocus:%.2f' % (cycle, zfocus))
                f.write('cycle:%02d\tzfocus:%.2f\n' % (cycle, zfocus))

                C.sendandrecv({"CCP": "CAM SET 2 %.3f" % BF_expTime})
                C.sendandrecv({"CCP": "LED_G OPEN"})
                #focus_z=C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 0 0 0"})  # 返回焦点数
                C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 0 0 0"})
                data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
                #img = com.data2image(data)
                img = com.data2image(data, [2048, 2048])
                imgcrop = img[624:1424, 624:1424]
                cv2.imwrite(
                    os.path.join(localfolder, 'img_cycle_%02d-BF_z%.2fum.tiff' % (cycle, zfocus)), imgcrop)
                C.sendandrecv({"CCP": "LED_G CLOSE"})

                C.sendandrecv({"CCP": "CAM SET 2 %.3f" % FF_expTime})
                C.sendandrecv({"CCP": "LED_B OPEN"})
                C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 0 0 0"})
                #             data = C.sendandrecv({"CCP": "CAM_CAPTURER TRIGGERPHOTO 0 0 0"})
                data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
                img = com.data2image(data, [2048, 2048])
                imgcrop = img[624:1424, 624:1424]
                fvB = calculation_module.SML(img)
                cv2.imwrite(
                    os.path.join(localfolder, 'img_cycle_%02d-FF_z%.2fum.tiff' % (cycle, zfocus)), img)
                C.sendandrecv({"CCP": "LED_B CLOSE"})
                #time.sleep(0.02)

                f.write('z\t G_WAV\t G_SML\t B_WAV\t B_SML\n')
                #C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 0 TIMEOUT 1000"})
                if fvB<0.2:
                    C.sendandrecv({"CCP": "WDI SET 4 %f TIMEOUT 1000" % (zfocus+45)})
                    C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 1 TIMEOUT 1000"})  # AutoFocus On!
                    time.sleep(0.5)
                    C.sendandrecv({"CCP": "WDI GET 5 TIMEOUT 1000"})
                    zData = C.sendandrecv({"CCP": "WDI GET 4 TIMEOUT 1000"})
                    # print('wdifocuse=%fum' % zData[b'data'][0])
                    z = format(zData[b'data'][0], '.3f')
                    zfocus = float(z)
                    C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 0 TIMEOUT 1000"})  # AutoFocus Off!


                for z in np.arange(zfocus - zRange, zfocus + zRange + zStep, zStep):
                    C.sendandrecv({"CCP": "WDI SET 4 %f TIMEOUT 1000" % z})  # 运动到z
                    time.sleep(0.03)
                    # C.sendandrecv({"CCP": "CAM SET 2 0.002"})
                    # C.sendandrecv({"CCP": "LED_G OPEN"})
                    # C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 0 0 0"})
                    #
                    # data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
                    # img = com.data2image(data)
                    # C.sendandrecv({"CCP": "LED_G CLOSE"})
                    # imgsmall = img[900:1100, 900:1100]
                    # imgcrop = img[624:1424, 624:1424]
                    # fvG_WT = calculation_module.WAVV(imgsmall)  # 反应照片清晰度的对焦值，小波变换
                    # fvG_ML = calculation_module.SML(imgsmall)  # 反应照片清晰度的对焦值，改进的拉普拉斯算子
                    #cv2.imwrite(os.path.join(imgpath, 'BFimg_cycle_%02d_z%.2fum_fv%.3f.tiff' % (
                    #    cycle, z, fvG_WT)), imgcrop)

                    C.sendandrecv({"CCP": "CAM SET 2 0.03"})
                    C.sendandrecv({"CCP": "LED_B OPEN"})
                    C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 0 0 0"})
                    #                 data = C.sendandrecv({"CCP": "CAM_CAPTURER TRIGGERPHOTO 0 0 0"})
                    data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
                    img = com.data2image(data)
                    C.sendandrecv({"CCP": "LED_B CLOSE"})
                    imgsmall = img[900:1100, 900:1100]
                    imgcrop = img[624:1424, 624:1424]
                    fvB_WT = calculation_module.WAVV(imgsmall)  # 反应照片清晰度的对焦值，小波变换
                    fvB_ML = calculation_module.SML(imgsmall)  # 反应照片清晰度的对焦值，改进的拉普拉斯算子
                    #                 print('pdB=%.3f\t fvB_WT=%.3f\t fvB_ML=%.3f' %(data[b'data'][0],fvB_WT,fvB_ML))
                    cv2.imwrite(os.path.join(imgpath, 'FFimg_cycle_%02d_z%.2fum_fv%.3f.tiff' % (
                        cycle, z, fvB_ML)), img)

                    #time.sleep(0.02)
                    fvG_WT=0.5
                    fvG_ML=0.5
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


    def WDI_ligt(self):

        greenLEDcurrent = float(self.Brightfield_value.text())
        blueLEDcurrent = float(self.fluorsecent_value.text())
        C.sendandrecv({"msgID": 1, "CCP": "LED_G SET 1 %.1f" % greenLEDcurrent})
        C.sendandrecv({"msgID": 1, "CCP": "LED_B SET 1 %.1f" % blueLEDcurrent})
        C.sendandrecv({"CCP": "CAM SET 3 0 0 2048 2048"})
        localtime0 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        name = localtime0
        localfolder = com.setOutputFolder(os.path.join('output\WDI_ligt' + '/' + name))
        FF_expTime = float(self.exposure_time.text())
        for cycle in range(10):
            C.sendandrecv({"CCP": "CAM SET 2 0.002"})
            C.sendandrecv({"CCP": "LED_G OPEN"})
            C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 0 0 0"})
            data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
            img = com.data2image(data)
            C.sendandrecv({"CCP": "LED_G CLOSE"})
            cv2.imwrite(os.path.join(localfolder, 'BFimg_%d.tiff' % (
            cycle)), img)
            C.sendandrecv({"CCP": "CAM SET 2 %.3f" % FF_expTime})
            C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 0 0 0"})
            data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
            img = com.data2image(data)
            cv2.imwrite(os.path.join(localfolder, 'WDI_img_%d.tiff' % (
                cycle)), img)



    def Field_curvature(self):
        self.close_all
        zData = C.sendandrecv({"CCP": "WDI GET 4 TIMEOUT 1000"})
        # print('wdifocuse=%fum' % zData[b'data'][0])
        z = format(zData[b'data'][0], '.3f')
        zfocus = float(z)
        zfocus=round(zfocus,2)
        greenLEDcurrent = float(self.Brightfield_value.text())
        blueLEDcurrent = float(self.fluorsecent_value.text())
        C.sendandrecv({"msgID": 1, "CCP": "LED_G SET 1 %.1f" % greenLEDcurrent})
        C.sendandrecv({"msgID": 1, "CCP": "LED_B SET 1 %.1f" % blueLEDcurrent})
        C.sendandrecv({"CCP": "CAM SET 3 0 0 2048 2048"})

        zRange = 5
        zStep = 0.2  # um
        localtime0 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        name = localtime0
        localfolder = com.setOutputFolder(os.path.join('output\ZField_curvature' + '/' + name))
        file = os.path.join(localfolder, name + '.txt')
        f = open(file, 'a')
        f.write('Start test\tAt:%s\tzRange=%f\tzStep=%f\n' % (localtime0, zRange, zStep))
        f.close()
        a = zfocus - zRange
        b = zfocus + zRange
        z_sanlist = np.arange(a, b, zStep)
        SML = np.zeros((len(z_sanlist), 16, 16))
        C.sendandrecv({"CCP": "CAM SET 3 0 0 2048 2048"})

        C.sendandrecv({"CCP": "WDI SET 4 %f TIMEOUT 1000" % zfocus})  # 运动到z


        time.sleep(0.02)
        count=0

        for z in z_sanlist:
            C.sendandrecv({"CCP": "WDI SET 4 %f TIMEOUT 1000" % z})  # 运动到z
            time.sleep(0.3)
            C.sendandrecv({"CCP": "CAM SET 2 0.03"})
            C.sendandrecv({"CCP": "LED_B OPEN"})
            C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 0 0 0"})
            data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
            img = com.data2image(data)
            C.sendandrecv({"CCP": "LED_B CLOSE"})
            cv2.imwrite(os.path.join(localfolder, 'FFimg_z%.2fum.tiff' % (z)), img)

            for j in range(16):
                for k in range(16):
                    subimg = img[j * 128:j * 128 + 128, k * 128:k * 128 + 128]
                    SMLS = calculation_module.SML(subimg)
                    SML[count, j, k] = SMLS
            center_fv = max(SML[:, 7, 7])
            count = count + 1

        Zs = np.zeros((16, 16))
        fv_z = SML[:, 7, 7]
        idx = np.where(fv_z == np.amax(center_fv))
        center_z = ((z_sanlist[idx]))


        for j in range(16):
            for k in range(16):
                fv_z = SML[:, j, k]
                idx = np.where(fv_z == np.amax(fv_z))
                Zs[j, k] = ((z_sanlist[idx])-center_z)




        I_left = Zs[range(2, 15), 0]
        I_right = Zs[15, range(2, 15)]
        I_up = Zs[0, range(2, 15)]
        I_down = Zs[15, range(2, 15)]
        I_DU = -np.mean(I_left) + np.mean(I_right)
        I_LR = -np.mean(I_down) + np.mean(I_up)

        plt.figure(figsize=(5, 5))
        plt.imshow(Zs, cmap='jet')
        plt.colorbar()
        plt.xlabel('Tile_X')
        plt.ylabel('Tile_Y')
        plt.title('Zmax=%.3f, Zmin=%.3f\nLeft-Right: %.2fum\nDown-Up: %.2fum' % (np.amax(Zs), np.amin(Zs), I_LR, I_DU))

        output_field_curvature = os.path.join(localfolder, 'field_curvature.png')
        plt.savefig(os.path.join(output_field_curvature))
        f = open(file, 'a')

        f.write ('Zmax=%.3f, Zmin=%.3f\nLeft-Right: %.2fum\nDown-Up: %.2fum\n' % (np.amax(Zs), np.amin(Zs), I_LR, I_DU))

        n=16
        nums = Zs

        for i in range(n):
            for j in range(n):
                f.write(str(round(Zs[i][j],2))+'\t')
            f.write('\n')


        f.close()




    def measure_z_map(self):
        self.close_all()
        global tilenumber
        FMfile = 'save/FM518_S.txt'
        FM, FMZ = com.readFocusMap(FMfile)
        tilenumber = len(FM)
        tilemap = com.TileMap('save/TM518.txt')
        C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 1 TIMEOUT 1000"})  # AutoFocus On!
        C.sendandrecv({"CCP": "SERVO01 MOV 4 %.2f 0"})
        C.sendandrecv({"CCP": "SERVO02 MOV 4 %.2f 0"})
        wait_time = float(0.05)
        time.sleep(wait_time)
        localtime0 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        move_localfolder = com.setOutputFolder('output\measure_z_map\\' + localtime0)
        zfocus_wdi_tile = []
        arry_z_tile=[]
        f = open(os.path.join(move_localfolder, 'z_map.txt'), 'a')


        Z_WDI = np.zeros((37, 14))
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
            zData = C.sendandrecv({"CCP": "WDI GET 4 TIMEOUT 1000"})

            z = format(zData[b'data'][0], '.3f')
            zfocus = float(z)

            zfocus_wdi_tile.append(zfocus)

            C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 0 TIMEOUT 1000"})  # AutoFocus Off!
            f.write('tile= %.3d\t Z=%.3f\n' % (tid,zfocus))

        for n in range(14):
            if n % 2 == 0:
                for m in range(37):
                    arry_z_tile.append(zfocus_wdi_tile[m+n*37])
            else:
                for m in range(36,-1,-1):
                    arry_z_tile.append(zfocus_wdi_tile[m+n*37])
        arry_z_tile = numpy.array(arry_z_tile)
        arry_z_tile=np.reshape(arry_z_tile,[14,37])
        zfocus_Image=np.transpose(arry_z_tile)


        max_z_value=max(zfocus_wdi_tile)
        min_z_value=min(zfocus_wdi_tile)

        f.write('max_z_value= %.3f\t min_z_value=%.3f\n' % (max_z_value,min_z_value))
        f.write('pv_z= %.3f\n' % (max_z_value-min_z_value))
        left_z = np.mean(zfocus_Image[0:36, 0])
        right_z = np.mean(zfocus_Image[0:36, 13])
        up_z = np.mean(zfocus_Image[0, 0:13])
        down_z = np.mean(zfocus_Image[36, 0:13])
        f.write('left-rigt= %.3f\n' % (left_z - right_z))
        f.write('up-down= %.3f\n' % (up_z - down_z))

        plt.figure(figsize=(5, 10))
        plt.imshow(zfocus_Image, cmap='jet')
        plt.colorbar()
        plt.xlabel('Tile_X')
        plt.ylabel('Tile_Y')
        plt.title('Z_MAP\n max_z_value= %.3f\t min_z_value=%.3f\n left-rigt= %.3f\n up-down= %.3f\n pv_z= %.3f\n' % (max_z_value, min_z_value ,(left_z - right_z),(up_z - down_z),(max_z_value-min_z_value)))
        output_shading_plot = os.path.join(move_localfolder, 'z_map_2.png')
        plt.savefig(os.path.join(output_shading_plot))
        plt.show()
        f.close()

    def static_focus(self):
        self.close_all()
        move_z = 5
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
            repeat_times = 5
            zfocus_up_all = []
            zfocus_down_all = []
            fv_up_all = []
            fv_down_all = []
            list_uniformity=[]
            for i in range(1, repeat_times + 1):
                with open(file, 'a') as f:
                    C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 1 TIMEOUT 1000"})  # AutoFocus On!
                    zData = C.sendandrecv({"CCP": "WDI GET 5 TIMEOUT 1000"})  # To Wait for FocusOver and Get z
                    zData_up = float(zData[b'data'][0])
                    time.sleep(1)
                    C.sendandrecv({"CCP": "CAM SET 2 %.3f" % 0.002})
                    C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 0 0 0"})
                    data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
                    img = com.data2image(data, [256, 256])
                    output_img = os.path.join(localfolder, 'tile_%03d_up_%03d.tiff' % (tid, i))
                    cv2.imwrite(output_img, img)
                    fv_midd_up = calculation_module.WAVV(img)
                    zfocus_up_all = np.append(zfocus_up_all, zData_up)
                    fv_up_all = np.append(fv_up_all, fv_midd_up)
                    C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 0 TIMEOUT 1000"})  # AutoFocus Off!
                    C.sendandrecv({"CCP": "WDI SET 4 %f TIMEOUT 1000" % (zData_up + move_z)})
                    time.sleep(1)
                    f.write('tid：%d from up: cyc:%d \tzfocus:%.3f \tfv_value:%.3f \n' % (tid, i, zData_up, fv_midd_up))
            uniformity = max(zfocus_up_all) - min(zfocus_up_all)
            list_uniformity= np.append(list_uniformity, uniformity)
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
                    img = com.data2image(data, [256, 256])
                    output_img = os.path.join(localfolder, 'tile_%03d_down_%03d.tiff' % (tid, i))
                    cv2.imwrite(output_img, img)
                    # fv_upleft=WAVV(img[20:220,20:220])
                    # fv_upright=WAVV(img[20:220,1828:2028])
                    fv_midd_down = calculation_module.WAVV(img)
                    zfocus_down_all = np.append(zfocus_down_all, zData_down)
                    fv_down_all = np.append(fv_down_all, fv_midd_down)
                    C.sendandrecv({"CCP": "WDI AUTOFOCUSCONTROL 0 TIMEOUT 1000"})  # AutoFocus Off!
                    C.sendandrecv({"CCP": "WDI SET 4 %f TIMEOUT 1000" % (zData_down - move_z)})
                    time.sleep(1)
                    f.write('tid：%d from down: cyc:%d \tzfocus:%.3f \tfv_value:%.3f \n' % (
                    tid, i, zData_down, fv_midd_down))
            uniformity = max(fv_down_all) - min(fv_down_all)
            list_uniformity = np.append(list_uniformity, uniformity)
            with open(file, 'a') as f:
                f.write('tid：%d from up:  \tuniformity:%.3f \n' % (tid, uniformity))
            C.sendandrecv({"CCP": "LED_G CLOSE"})
            f = open(file, 'a')
            localtime = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
            f.write('%s \tFinish!\tAt:%s\n' % (name, localtime))


        max_unitformit=np.max(list_uniformity)
        min_unitformit = np.min(list_uniformity)
        mean_unitformit = np.mean(list_uniformity)
        f.write('max:%.3f\tmin:%.3f\tmean:%.3f\t' % (max_unitformit, min_unitformit,mean_unitformit))
        f.close()
        self.textBrowser.clear()
        self.textBrowser.setText('对焦测试结束\n')
        self.textBrowser.setText('max:%.3f\tmin:%.3f\tmean:%.3f\n' % (max_unitformit, min_unitformit,mean_unitformit))



    def means_astigmatism(self):
        self.close_all()
        list_fvB_ML = []
        zData = C.sendandrecv({"CCP": "WDI GET 4 TIMEOUT 1000"})
        # print('wdifocuse=%fum' % zData[b'data'][0])
        z = format(zData[b'data'][0], '.3f')
        zfocus = float(z)
        blueLEDcurrent = 13
        C.sendandrecv({"msgID": 1, "CCP": "LED_B SET 1 %.1f" % blueLEDcurrent})
        C.sendandrecv({"CCP": "CAM SET 3 0 0 2048 2048"})
        zRange = 8
        zStep = 0.2  # um
        localtime0 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        name = localtime0
        outputfolder = 'output\z_get_imge'
        localfolder = com.setOutputFolder(os.path.join('output\means_astigmatism' + '/' + name))

        imgpath = localfolder
        if not os.path.exists(imgpath):
            os.mkdir(imgpath)
        for z in np.arange(zfocus - zRange, zfocus + zRange + zStep, zStep):
            C.sendandrecv({"CCP": "WDI SET 4 %f TIMEOUT 1000" % z})  # 运动到z
            time.sleep(0.02)
            C.sendandrecv({"CCP": "CAM SET 2 0.1"})
            C.sendandrecv({"CCP": "LED_B OPEN"})
            C.sendandrecv({"CCP": "WDI_CAPTURER TRIGGERPHOTO 0 0 0"})
            data = C.sendandrecv({"CCP": "CAM GETIMAGE"})
            img = com.data2image(data)
            C.sendandrecv({"CCP": "LED_B CLOSE"})
            imgsmall = img[900:1100, 900:1100]
            #               fvB_WT = com.WAVV(imgsmall)  # 反应照片清晰度的对焦值，小波变换
            fvB_ML = calculation_module.SML(imgsmall)  # 反应照片清晰度的对焦值，改进的拉普拉斯算子
            list_fvB_ML = np.append(list_fvB_ML, fvB_ML)
            cv2.imwrite(os.path.join(imgpath, 'FFimg_z%.2fum_fv%.3f.tiff' % (
                z, fvB_ML)), img)
            time.sleep(0.02)
        self.textBrowser.clear()
        self.textBrowser.setText('像散测试结束')
        #        max_ind=list_fvB_ML.index(max(list_fvB_ML))
        #        getz=zfocus - zRange+max_ind*0.2+1

    def get_resolution(self, file_pathname):
        Modulation_x_list = np.zeros((200, 9))
        max_value_x = np.zeros((1, 9))
        Modulation_y_list = np.zeros((200, 9))
        max_value_y = np.zeros((1, 9))
        j = 0

        for filename in os.listdir(file_pathname):
            # print(filename)
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
                center, image_8bit = calculation_module.cal_center(image_rio[:, :, i])
                Modulation_x, Modulation_y = calculation_module.cal_Modulation(center, image_8bit)
                if (np.isnan(Modulation_x)):
                    Modulation_x = 0
                if (np.isnan(Modulation_y)):
                    Modulation_y = 0
                Modulation_x_list[j, i] = Modulation_x
                Modulation_y_list[j, i] = Modulation_y
            j = j + 1

        file = os.path.join(file_pathname + '\\' + 'resolution.txt')
        f = open(file, 'a')
        self.textBrowser.clear()
        self.textBrowser.setText('分辨力测试结束')
        # max_value_x = max(Modulation_x_list[:, 0])
        index_x = np.argmax(Modulation_x_list[:, 0])
        # max_value_y = max(Modulation_x_list[:, 0])
        index_y = np.argmax(Modulation_y_list[:, 0])

        for i in range(9):
            max_value_x[0, i] = max(Modulation_x_list[:, i])
            max_value_y[0, i] = max(Modulation_y_list[:, i])
            f.write('cross:%.3f   column:%.3f \n' % (max_value_x[0, i], max_value_y[0, i]))
            self.textBrowser.append('cross:%.3f   column:%.3f' % (max_value_x[0, i], max_value_y[0, i]))
        f.close()




    def measure_resolution(self):
        self.close_all()
        zData = C.sendandrecv({"CCP": "WDI GET 4 TIMEOUT 1000"})
        # print('wdifocuse=%fum' % zData[b'data'][0])
        z = format(zData[b'data'][0], '.3f')
        zfocus = float(z)
        #blueLEDcurrent = 10
        FFLEDcurrent = float(self.fluorsecent_value.text())
        C.sendandrecv({"msgID": 1, "CCP": "LED_B SET 1 %.1f" % FFLEDcurrent})
        C.sendandrecv({"CCP": "CAM SET 3 0 0 2048 2048"})
        zRange = 5
        zStep = 0.2  # um
        localtime0 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        name = localtime0
        localfolder = com.setOutputFolder(os.path.join('output\measure_resolution' + '/' + name))

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



    def measure_rotation(self):
        self.close_all()
        C.sendandrecv({"msgID": 1, "CCP": "LED_G SET 1 %.3f" % 0.1})
        C.sendandrecv({"CCP": "LED_G OPEN"})
        localtime0 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        name = localtime0
        localfolder = com.setOutputFolder(os.path.join('output\scal_rotation' + '/' + name))
        output_img = os.path.join(localfolder, 'cal_rotation.tiff')
        expTime = float(self.exposure_time.text())
        image = self.cap_image(expTime, 1, output_img)
        C.sendandrecv({"CCP": "LED_G CLOSE"})
        # image = image[200:1800, 1024 - 700:1024 + 700]
        mean_delta=calculation_module.cal_rota(image)

        self.textBrowser.clear()
        self.textBrowser.setText('相机旋转角度\n')
        self.textBrowser.setText('偏离%.4f像素  ' % (mean_delta))
        file = os.path.join(localfolder, name + '.txt')
        f = open(file, 'a')
        f.write('偏离像素： %.3f \n' % (mean_delta))
        f.close()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", " Optical test V 0.1.7"))
        self.chip_in_button.setText(_translate("MainWindow", "进仓"))
        self.chip_out_button.setText(_translate("MainWindow", "出仓"))
        self.z_move_button.setText(_translate("MainWindow", "Z"))
        self.x_move_button.setText(_translate("MainWindow", "X"))
        self.y_move_button.setText(_translate("MainWindow", "Y"))
        self.close_button.setText(_translate("MainWindow", "关闭"))
        self.x_reset_button.setText(_translate("MainWindow", "X复位"))
        self.y_reset_button.setText(_translate("MainWindow", "Y复位"))
        self.fouce_on_button.setText(_translate("MainWindow", "开启WDI对焦"))
        self.make_0_button.setText(_translate("MainWindow", "make 0"))
        self.y_positive_button.setText(_translate("MainWindow", "+"))
        self.y_negative_button.setText(_translate("MainWindow", "-"))
        self.x_negative_button.setText(_translate("MainWindow", "-"))
        self.x_positive_button.setText(_translate("MainWindow", "+"))
        self.z_up_button.setText(_translate("MainWindow", "+"))
        self.z_down_button.setText(_translate("MainWindow", "-"))
        self.fluorsecent_on_button.setText(_translate("MainWindow", "开启"))
        self.Brightfield_on_button.setText(_translate("MainWindow", "开启"))
        self.fluorsecent_off_button.setText(_translate("MainWindow", "关闭"))
        self.Brightfield_off_button.setText(_translate("MainWindow", "关闭"))
        self.label_flur.setText(_translate("MainWindow", "荧光"))
        self.label_Bright.setText(_translate("MainWindow", "明场"))
        self.label_wdi.setText(_translate("MainWindow", "WDI"))
        self.bright_capture_button.setText(_translate("MainWindow", "单帧采集"))
        self.fouce_off_buttion.setText(_translate("MainWindow", "关闭WDI对焦"))
        self.zscan_button.setText(_translate("MainWindow", "z_scan"))
        self.label_wdi_up.setText(_translate("MainWindow", "上"))
        self.label_wdi_down.setText(_translate("MainWindow", "下"))
        self.label_wdi_step.setText(_translate("MainWindow", "步长"))
        self.continuous_capture_button_on.setText(_translate("MainWindow", "连续采集开启"))
        self.continuous_capture_button_off.setText(_translate("MainWindow", "连续采集关闭"))
        self.exposure_time_label.setText(_translate("MainWindow", "曝光时间"))
        self.shading_button.setText(_translate("MainWindow", "照明均匀度"))
        self.Resolution_button.setText(_translate("MainWindow", "分辨率"))
        self.flat_button.setText(_translate("MainWindow", "调平评价"))
        self.Z_MAP_button.setText(_translate("MainWindow", "Z_MAP"))
        self.Field_curvature_button.setText(_translate("MainWindow", "场曲"))

        self.Nine_View_button.setText(_translate("MainWindow", "九宫格显示"))
        self.Background_light_test_button.setText(_translate("MainWindow", "背景杂光测试"))
        self.cal_rotation_button.setText(_translate("MainWindow", "相机旋转角度"))
        self.WDI_test_button.setText(_translate("MainWindow", "对焦精度测量"))
        self.static_focus_button.setText(_translate("MainWindow", "重复对焦精度"))
        self.astigmatism_button.setText(_translate("MainWindow", "像散测试"))








