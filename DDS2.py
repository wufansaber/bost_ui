import zmq
import msgpack
import time
import logging, os, sys, time
from datetime import datetime
import threading
import queue
import numpy as np
import clr
import sys
sys.path.append(r'./')  # 加载c# dll文件路径
clr.FindAssembly("ZMQDS")  # 加载动态库
from ZMQDS import *  # 引入命名空间


version = 1.1
# 通过message ID判断命令是否返回，20181210
version = 3.0
# 20190213：
# update CCPv0.4 to CCPv1.0
# update zmq msg serialization struct
# add init_imgreceiver
# change capture function:IMGDATA %d %d %d %d
# change responseCode 0 -> 0x00000000
version = 4.0
# support COMPASS-DS v2.0
version = 4.1
# modify send_msg func: delete msg format checkout.

instance=InterFace   # 类名
instance.Init()  # 端口初始化
# instance.DllTest()


def createLogger(outputfolder, level=logging.INFO):
    rq = time.strftime('NSDS-%Y%m%d%H%M', time.localtime(time.time()))
    logfile = os.path.join(outputfolder, rq + '.log')

    logger = logging.getLogger('NSDS')
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])
    logger.setLevel(level)
    # create a handler, write the log info into it
    fh = logging.FileHandler(logfile)
    fh.setLevel(level)
    # create another handler output the log though console
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    # 定义handler的输出格式
    formatter = logging.Formatter(" %(relativeCreated)d - %(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    # 记录一条日志
    logger.info('New Log begins:{}'.format(logfile))
    return logger


class NSDS(object):
    MSG = {}

    def __init__(self, logpath='./', loglevel=logging.DEBUG):
        self.AUTORECV = False
        # self.Q=queue.Queue()
        self.msgIDcounter = 0
        self.context = zmq.Context()
        self.logger = createLogger(logpath, level=loglevel)
        # self._init_cam()

    def _init_cam(self):
        self.roi = (0, 0, 2048, 2048)
        self.setROI((0, 0, 2048, 2048), force=True)
        self.exptime = 0.02
        self.setExptime(0.02, force=True)

    def _init_LED(self):
        self.LEDstatus = {'LED_G': False, 'LED_B': False}
        self.LED('b', False, force=True)
        self.LED('g', False, force=True)

    # v4.0, 通讯接口初始化拿掉。改为直接调用DS_dll。
    def init_msgclient(self, ip='127.0.0.1', port=5555):
        pass

    def init_datreceiver(self, ip='*', port=5577):
        pass

    def init_imgreceiver(self, ip='*', port=5577):
        pass

    def send_msg(self, msg=None, msg_dict=None, sender=None):
        # if isinstance(msg, dict) and 'MsgID' in msg:
        send_ccp_str = msg['CCP']
        self.logger.debug('send_msg:{}'.format(send_ccp_str))

        if 'CAM GETIMAGE' in send_ccp_str:
            data = instance.GetImage()  # 获取图像
            self.logger.debug('<== receive_msg: msg_id: {} error_code: {}'.format(data['MsgID'], data['ErrorCode']))
            if data['ErrorCode'] != '0x00000000':
                self.logger.error('receive_msg: msg_id: {} error_code: {}'.format(data['MsgID'], data['ErrorCode']))
        else:
            data = instance.SendCCP(send_ccp_str)  # 纯发送CCP
            self.logger.debug('receive_msg: msg_id: {} error_code: {}'.format(data['MsgID'], data['ErrorCode']))
            if data['ErrorCode'] != '0x00000000':
                self.logger.error('receive_msg: msg_id: {} error_code: {}'.format(data['MsgID'], data['ErrorCode']))

    def sendandrecv(self, msg, timeout=5, retry_limit=3):
        send_ccp_str = msg['CCP']
        self.logger.debug('send_msg:{}'.format(send_ccp_str))
        data1 = {}
        if 'CAM GETIMAGE' in send_ccp_str:
            data = instance.GetImage()   # 获取图像
            data1['time'] = time.time()
            #data1[b'data'] = data['Data'].encode("utf-8")
            f=open(data['Data'],'rb')
            imageData=f.read()
            f.close()
            data1[b'data']= imageData
            data1['MsgID'] = data['MsgID']
            data1['ErrorCode'] = data['ErrorCode']
            self.logger.debug('receive_msg: msg_id: {} error_code: {}'.format(data['MsgID'], data['ErrorCode']))
            if data['ErrorCode'] != '0x00000000':
                self.logger.error('receive_msg: msg_id: {} error_code: {}'.format(data['MsgID'], data['ErrorCode']))
        else:
            data = instance.SendRecCCP(send_ccp_str)  # 发送CCP并接收其返回结果
            data1['time'] = time.time()
            data1[b'data'] = data['Data']
            data1['MsgID'] = data['MsgID']
            data1['ErrorCode'] = data['ErrorCode']
            self.logger.debug('receive_msg: msg_id: {} error_code: {}'.format(data['MsgID'], data['ErrorCode']))
            if data['ErrorCode'] != '0x00000000':
                self.logger.error('receive_msg: msg_id: {} error_code: {}'.format(data['MsgID'], data['ErrorCode']))
        return data1

    def data2image(self, data, shape):
        raw = np.frombuffer(data['Data'], np.uint16)
        if shape[0] * shape[1] == len(raw):
            image = np.reshape(raw, shape)
        else:
            image = raw
        return image

    def LED(self, led, newstatus, force=False):
        if led == 'b' or led == 'blue' or led == 0:
            ledstring = 'LED_B'
        elif led == 'g' or led == 'green' or led == 1:
            ledstring = 'LED_G'

        if self.LEDstatus[ledstring] == newstatus and not force:
            self.logger.debug('Skip LED setting!')
            return

        if newstatus:
            cmd = 'OPEN'
        else:
            cmd = 'CLOSE'

        ret = self.sendandrecv({"CCP": "%s %s" % (ledstring, cmd)})

        if ret['ErrorCode'] == '0x00000000':  # 兼容DS故障编码规则
            self.LEDstatus[ledstring] = newstatus
        else:
            self.logger.error('Setting LED error! %s %s' % (ledstring, cmd))

    def setROI(self, newroi, force=False):
        if self.roi == newroi and not force:
            self.logger.debug('Skip ROI setting! Old:{} New:{}'.format(self.roi, newroi))
            return
        ret = self.sendandrecv({"CCP": "CAM SET 3 %d %d %d %d" % tuple(newroi)})
        if ret['ErrorCode'] == '0x00000000':  # 兼容DS故障编码规则
            self.roi = newroi
        else:
            self.logger.error('Setting ROI error! Old:{} New:{}'.format(self.roi, newroi))

    def setExptime(self, newexptime, force=False):
        if self.exptime == newexptime and not force:
            self.logger.debug('Skip Exposure Time setting! Old:{} New:{}'.format(self.exptime, newexptime))
            return
        ret = self.sendandrecv({"CCP": "CAM SET 2 %f" % (newexptime)})
        if ret['ErrorCode'] == '0x00000000':  # 兼容DS故障编码规则
            self.exptime = newexptime
        else:
            self.logger.error('Setting Exposure Time error! Old:{} New:{}'.format(self.exptime, newexptime))

    def capture(self):
        retryCount = 0
        while retryCount < 5:
            t1 = time.time()
            data = self.sendandrecv({"CCP": "CAM_CAPTURER TRIGGERPHOTO 0 0 0"})
            if not data['ErrorCode'] == '0x00000000':  # 兼容DS故障编码规则
                self.logger.error('CAPTURE error! {} '.format(data))

            t2 = time.time()
            data = self.sendandrecv({"CCP": "CAM GETIMAGE"})
            if not data['ErrorCode'] == '0x00000000':  # 兼容DS故障编码规则
                # self.logger.error('GETIMAGE error! {} '.format(data))
                self.logger.error('GETIMAGE error!')
            t3 = time.time()
            img = self.data2image(data, shape=list(self.roi[2:4]))

            if img.mean() > 90 and img.shape == tuple(self.roi[2:4]):
                return img
            else:
                self.logger.error('Got bad image! Retrying.. %d' % retryCount)
                self.setROI(self.roi, force=True)
                retryCount += 1
                continue
        self.logger.error('Fail capture, try 5 times !')

    def capture_old(self):
        retryCount = 0
        while retryCount < 5:
            t1 = time.time()
            data = self.sendandrecv({"CCP": "CAM_CAPTURER TRIGGERPHOTO 0 0 0"})
            if not data['ErrorCode'] == '0x00000000':  # 兼容DS故障编码规则
                self.logger.error('CAPTURE error! {} '.format(data))

            t2 = time.time()
            data = self.sendandrecv({"CCP": "CAM GETIMAGE"})
            if not data['ErrorCode'] == '0x00000000':  # 兼容DS故障编码规则
                # self.logger.error('GETIMAGE error! {} '.format(data))
                self.logger.error('GETIMAGE error!')
            t3 = time.time()

            img = self.data2image(data, shape=[2048, 2048])

            if img.mean() > 90 and img.shape == tuple([2048, 2048]):
                return img
            else:
                self.logger.error('Got bad image! Retrying.. %d' % retryCount)
                retryCount += 1
                continue
        self.logger.error('Fail capture, try 5 times !')

    # ================ backup =============
    def sendandrecv2(self, msg, timeout=600):
        if not self.AUTORECV:
            print('Auto receive is off, please use sendandrecv(msg)')
            return None
        self.msgIDcounter += 1
        msg['MsgID'] = self.msgIDcounter
        self.send_msg(msg)
        t = 0
        while (self.msgIDcounter not in self.MSG.keys()):
            time.sleep(0.1)
            t += 0.1
            if t > timeout:
                break
        if self.msgIDcounter in self.MSG.keys():
            data = self.MSG[self.msgIDcounter]
        else:
            data = None
        return data

    def send_msg2(self, msg=None, msg_dict=None, sender=None):
        self.msgIDcounter += 1
        msg['MsgID'] = self.msgIDcounter
        self.send_msg(msg)
        return self.msgIDcounter