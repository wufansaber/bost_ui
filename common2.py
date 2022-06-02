
import logging, os, sys, time
import csv
import numpy as np
from matplotlib import pyplot as plt


def readFocusMap(filename):
    FM = []
    FMZ = []
    with open(filename) as f:
        spam = csv.reader(f, delimiter = '\t')
        for line in spam:
            if len(line) == 1:
                FM.append(int(line[0]))
            if len(line) == 2:
                FM.append(int(line[0]))
                FMZ.append( ( int(line[0]), float(line[1]) ) )
    return FM,FMZ

def readFocusMapAuto(filename):
    FMZ = readCSV(filename)
    return FMZ

def readCSV(filename):
    return np.genfromtxt(filename)

def writeFocusMap(filename,FMZ):
    with open(filename,'w') as f:
        spam = csv.writer(f, delimiter = '\t',quotechar='|',quoting = csv.QUOTE_MINIMAL)
        for line in FMZ:
            spam.writerow(list(line))


def readFastscan(filename):
    lst_fs=[]
    for line in open(filename).readlines():
        if line[0]=='>':
            pass
            tid_focus=int(line[1:])
        else:
            elements=[int(i) for i in line.split()]
            lst_fs.append((tid_focus,elements))
    return lst_fs


    # save



def readTileMap(filename):
    version = '1.1 rename function name into readTileMap'
    lst_tm=[]
    dct_tm={}
    dct_ij={}
    dct_param={}
    START_TM=False
    for line in open(filename).readlines():
        if line[0:5] == 'Param':
            params=[i.split('=') for i in line.split()[1:]]
            for param in params:
                if len(param)==2:
                    dct_param[param[0]]=float(param[1])
                else:
                    # raise error
                    pass
        if line[0:5] == '##TID':
            START_TM=True
            continue
        if START_TM:
            elements=[float(i) for i in line.split()]
            tid=int(elements[0])
            i=int(elements[1])
            j=int(elements[2])
            x=elements[3]
            y=elements[4]
            dct_tm[tid]=(tid,i,j,x,y)
            dct_ij[(i,j)]=(tid,i,j,x,y)
            lst_tm.append([float(i) for i in elements[1:]])
    return dct_tm,dct_ij,dct_param



def setOutputFolder(outputfolder):
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    return outputfolder

def createLogger(outputfolder):
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logfile = os.path.join(outputfolder , rq + '.log')

    logger = logging.getLogger('test_log')
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])

    logger.setLevel(logging.DEBUG)
    # create a handler, write the log info into it
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
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



def data2image(data,shape=[2048,2048]):
    image=np.reshape(np.frombuffer(data[b'data'], np.uint16),shape)
    return image

def image_16_8(self, image):
        min_16bit = np.min(image)
        max_16bit = np.max(image)
        # image_8bit = np.array(np.rint((255.0 * (image_16bit - min_16bit)) / float(max_16bit - min_16bit)), dtype=np.uint8)
        # 或者下面一种写法
        image_8bit = np.array(np.rint(255 * ((image - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
        return image_8bit




class TileMap():
    def __init__(self,TMfile):
        version = '1.1 rename function name into readTileMap'
        lst_tm=[]
        dct_tm={}
        dct_ji={}
        dct_param={}
        START_TM=False
        self.raw = np.ndarray([0,5])
        self._xoffset = 0
        self._yoffset = 0
        with open(TMfile) as f:
            for line in f.readlines():
                if line[0:5] == 'Param':
                    params=[i.split('=') for i in line.split()[1:]]
                    for param in params:
                        if len(param)==2:
                            dct_param[param[0]]=float(param[1])
                            if param[0]=='XOffset':
                                self._xoffset = dct_param[param[0]]
                            elif  param[0]=='YOffset':
                                self._yoffset = dct_param[param[0]]
                        else:
                            # raise error
                            pass

                if line[0:5] == '##TID':
                    START_TM=True
                    continue
                if START_TM:
                    max_i=0
                    max_j=0
                    elements=[float(i) for i in line.split()]
                    tid=int(elements[0])
                    i=int(elements[1])
                    j=int(elements[2])
                    if i>max_i:
                        max_i=i
                    if j>max_j:
                        max_j=j
                    x=elements[3]+self._xoffset
                    y=elements[4]+self._yoffset

                    self.raw = np.r_[self.raw,np.array((tid,i,j,x,y),ndmin = 2)]
                    dct_tm[tid]=(tid,i,j,x,y)
                    dct_ji[(j,i)]=(tid,i,j,x,y)
                    lst_tm.append([float(i) for i in elements[1:]])

        self.dTM = dct_tm
        self.dTMji = dct_ji
        self.param = dct_param
        self.shape = (max_j,max_i)
        self.shapeH = (max_i,max_j)
        self._offset = ( self._xoffset,self._yoffset )
        self.ntile = len(dct_tm)
        self.gi,self.gj = np.mgrid[1:self.shape[1]+1,1:self.shape[0]+1]

    def tilelist2array(self,tilelist):
        pass

    def ji2t(self,jis):
        ts = []

        for ji in jis:
            tempji=np.array(ji,dtype=int)
            ts.append(self.dTMji[tuple(tempji)][0])
        return np.array(ts)
        pass

    def t2ji(self,ts):
        idx=np.array([],dtype=np.int64)
        for t in ts:
        #             idx.append(np.where(self.raw[:,0] == t))
            idx = np.r_[ idx,np.where(self.raw[:,0] == t)[0] ]
        return  self.raw[idx,1:3]

    def t2xy(self,ts):
        idx=np.array([],dtype=np.int64)
        for t in ts:
        #             idx.append(np.where(self.raw[:,0] == t))
            idx = np.r_[ idx,np.where(self.raw[:,0] == t)[0] ]
        return  self.raw[idx,3:5]
    def grid2tilelist(self,grid):
        pass
#         output = np.ndarray([grid.size,2])

#         for i in grid.shape[1]:
#             for j in grid.shape[0]:
#                 output[self.ji2t([(j,i)]),:] = np.array([])

    def tilelist2grid(self,tilelist):
        pass

    def grid2tv(self,gv):
        tv = np.ndarray( (0,2) )
        for idx in range(0,self.ntile):
            j = self.gj.flat[idx]
            i = self.gi.flat[idx]
            t = self.dTMji[(j,i)][0]
            tv=np.r_[tv, np.array( [t, gv[j-1,i-1]], ndmin=2) ]
        return tv

    def FMZ2array(self,FMZ):

        gz=np.zeros(self.shape)

        for fmz in FMZ:
            tid = fmz[0]
            z= fmz[1]
            i,j=self.dTM[tid][1:3]
    #         print(tid,i,j)
            i-=1
            j-=1
            gz[j,i] = z
        return gz

