import os
import glob
import numpy as np
import sys
import re
from io import StringIO
import matplotlib.pyplot as plt
import math
import pickle

from outliers import smirnov_grubbs as grubbs


def Zscan_data_extract_singlechip(runID,result_path,save_path):
    chipdir=os.path.join(result_path,runID)
    print(chipdir)
    nameFolds=glob.glob(os.path.join(chipdir,'X*'))
    tileNum=len(nameFolds)
    print(len(nameFolds))
    if tileNum==0:
        print('No data in %s'%runID)
        sys.exit()
    if not os.path.exists(os.path.join(save_path,runID)):
        os.makedirs(os.path.join(save_path,runID))
    save_path1=os.path.join(save_path,runID)

    shift_info_wav=np.zeros((tileNum,3))
    shift_info_sml=np.zeros((tileNum,3))
    X=np.zeros((tileNum,1))
    Y=np.zeros((tileNum,1))
    Z=np.zeros((tileNum,1))
    Z_WDI=np.zeros((tileNum,1))
    Fv_max=np.zeros((tileNum,1))
    plt.figure(figsize=(35,20))
    for n in range(tileNum):
        fileName=nameFolds[n].split('\\')[-1]
        A=re.split(r'[XY,_]',fileName)
        X[n]=A[1]
        Y[n]=A[3]
        infoFile=os.path.join(nameFolds[n],'%s.txt'%fileName)
        with open(infoFile) as file:
            content=file.readlines()
            content=content[4:len(content)-2]
            C= np.loadtxt(content)
        if n<10:
            '''draw plot'''
            plt.subplot(5,4,2*(n+1)-1)
            plt.plot(C[:,0],C[:,1])
            plt.plot(C[:,0],C[:,3])
            max_value=np.max(C[:,3])
            max_pos=np.argmax(C[:,3],axis=0)
            plt.plot([C[:,0][max_pos],C[:,0][max_pos]],[0,max_value])
            Z[n]=C[:,0][max_pos]
            Fv_max[n]=max_value
            fv_shift1,fv_shift2,WDI_shift=F_getFvShift(C[:,1],C[:,3])
            plt.title('Fv_WAV_tile%d,shift:%.2f,%.2f,WDI_shift:%.2f'%(n,fv_shift1,fv_shift2,WDI_shift))
            shift_info_wav[n,:]=[fv_shift1,fv_shift2,WDI_shift]

            plt.subplot(5,4,2*(n+1))
            plt.plot(C[:,0],C[:,2])
            plt.plot(C[:,0],C[:,4])
            max_value=np.max(C[:,4])
            max_pos=np.argmax(C[:,4],axis=0)
            plt.plot([C[:,0][max_pos],C[:,0][max_pos]],[0,max_value])
            fv_shift1,fv_shift2,WDI_shift=F_getFvShift(C[:,2],C[:,4])
            plt.title('Fv_SML_tile%d,shift:%.2f,%.2f,WDI_shift:%.2f'%(n,fv_shift1,fv_shift2,WDI_shift))
            shift_info_sml[n,:]=[fv_shift1,fv_shift2,WDI_shift]
        max_value=np.max(C[:,3])
        max_pos=np.argmax(C[:,4],axis=0)
        Z[n]=C[:,0][max_pos]
        Z_WDI[n]=C[:,0][math.floor((len(C[:,0])+1)/2)]
        Fv_max[n]=max_value
        fv_shift1,fv_shift2,WDI_shift=F_getFvShift(C[:,1],C[:,3])
        shift_info_wav[n,:]=[fv_shift1,fv_shift2,WDI_shift]
        fv_shift1,fv_shift2,WDI_shift=F_getFvShift(C[:,2],C[:,4])
        shift_info_sml[n,:]=[fv_shift1,fv_shift2,WDI_shift]
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=None, hspace=0.15)
    plt.savefig(os.path.join(save_path1,'fv_curve_'+runID+'.png'),dpi=500,bbox_inches = 'tight')
    plt.close()
    #plt.show()

    '''shift info figure'''
    range_shift1 = np.ptp(shift_info_wav[:, 0])
    range_shift2 = np.ptp(shift_info_wav[:, 1])
    range_wdishift = np.ptp(shift_info_wav[:, 2])

    shift1 = shift_info_wav[:, 0]
    shift2 = shift_info_wav[:, 1]
    wdi_shift = shift_info_wav[:, 2]
    shift1 = grubbs.test(shift1, alpha=.05)
    shift2 = grubbs.test(shift2, alpha=.05)
    wdi_shift = grubbs.test(wdi_shift, alpha=.05)

    std_shift1 = np.std(shift1, ddof=1)
    std_shift2 = np.std(shift2, ddof=1)
    std_WDI = np.std(wdi_shift, ddof=1)
    mean_shift1 = np.mean(shift1)
    mean_shift2 = np.mean(shift2)
    mean_WDI = np.mean(wdi_shift)
    plt.figure(figsize=(20,10))
    plt.subplot(3,2,1)
    line1,=plt.plot(shift_info_wav[:,0],marker='o',markerfacecolor='none',markersize = '5',linestyle='-',linewidth='1')
    plt.legend(handles=[line1], labels=['shift1,range %s std %s mean %s'%(range_shift1,std_shift1,mean_shift1)], loc='best',fontsize=8)
    plt.title('WAV offset summary')

    plt.subplot(3,2,3)
    line2,=plt.plot(shift_info_wav[:,1],marker='*',markerfacecolor='none',markersize = '5',linestyle='-',linewidth='1')
    plt.legend(handles=[line2], labels=['shift2,range %s std %s mean %s'%(range_shift2,std_shift2,mean_shift2)], loc='best',fontsize=8)

    plt.subplot(3,2,5)
    line3,=plt.plot(shift_info_wav[:,2],marker='^',markerfacecolor='none',markersize = '5',linestyle='-',linewidth='1')
    plt.plot([0,tileNum],[mean_WDI,mean_WDI],'r',linewidth='0.5')
    plt.legend(handles=[line3], labels=['WDI_s,range %s std %s mean %s'%(range_wdishift,std_WDI,mean_WDI)], loc='best',fontsize=8)

    range_shift1 = np.ptp(shift_info_sml[:, 0])
    range_shift2 = np.ptp(shift_info_sml[:, 1])
    range_wdishift = np.ptp(shift_info_sml[:, 2])
    shift1 = shift_info_sml[:, 0]
    shift2 = shift_info_sml[:, 1]
    wdi_shift = shift_info_sml[:, 2]
    shift1 = grubbs.test(shift1, alpha=.05)
    shift2 = grubbs.test(shift2, alpha=.05)
    wdi_shift = grubbs.test(wdi_shift, alpha=.05)

    std_shift1 = np.std(shift1, ddof=1)
    std_shift2 = np.std(shift2, ddof=1)
    std_WDI = np.std(wdi_shift, ddof=1)
    mean_shift1 = np.mean(shift1)
    mean_shift2 = np.mean(shift2)
    mean_WDI = np.mean(wdi_shift)
    plt.subplot(3,2,2)
    line4,=plt.plot(shift_info_sml[:,0],marker='o',markerfacecolor='none',markersize = '5',linestyle='-',linewidth='1')
    plt.legend(handles=[line4], labels=['shift1,range %s std %s mean %s'%(range_shift1,std_shift1,mean_shift1)], loc='best',fontsize=8)
    plt.title('SML offset summary')

    plt.subplot(3,2,4)
    line5,=plt.plot(shift_info_sml[:,1],marker='*',markerfacecolor='none',markersize = '5',linestyle='-',linewidth='1')
    plt.legend(handles=[line5], labels=['shift2,range %s std %s mean %s'%(range_shift2,std_shift2,mean_shift2)], loc='best',fontsize=8)

    plt.subplot(3,2,6)
    line6,=plt.plot(shift_info_sml[:,2],marker='^',markerfacecolor='none',markersize = '5',linestyle='-',linewidth='1')
    plt.plot([0,tileNum],[mean_WDI,mean_WDI],'r',linewidth='0.5')
    plt.legend(handles=[line6], labels=['WDI_s,range %s std %s mean %s'%(range_wdishift,std_WDI,mean_WDI)], loc='best',fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=None, hspace=0.15)
    plt.savefig(os.path.join(save_path1,'offset_summary_'+runID+'.png'),dpi=500,bbox_inches = 'tight')
    plt.close()
    #plt.show()

    '''shift spatial figure'''
    plt.figure(figsize=(40,10))
    ax1 = plt.subplot(1,5,1)
    '''关闭裁剪功能'''
    plt.scatter(X,Y,s=300,c=Z,clip_on = False)
    plt.gca().set_aspect(1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    plt.colorbar()
    plt.title('focus_Z',fontsize=20)

    ax2 = plt.subplot(1,5,2)
    plt.scatter(X,Y,s=300,c=Z_WDI,clip_on = False)
    #XY轴单位长度相同
    plt.gca().set_aspect(1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    plt.colorbar()
    plt.title('WDI_Z',fontsize=20)

    ax3 = plt.subplot(1,5,3)
    c=np.squeeze(shift_info_sml[:,2])
    #C的值不能是列向量并且和XY保持一致
    plt.scatter(X,Y,s=300,c=shift_info_sml[:,2].reshape(len(X),1),clip_on = False)
    #XY轴单位长度相同
    plt.gca().set_aspect(1)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    plt.colorbar()
    plt.title('WDI_offset',fontsize=20)

    ax4 = plt.subplot(1,5,4)
    plt.scatter(X,Y,s=300,c=shift_info_sml[:,0].reshape(len(X),1),clip_on = False)
    #XY轴单位长度相同
    plt.gca().set_aspect(1)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    plt.colorbar()
    plt.title('shift1',fontsize=20)

    ax5 = plt.subplot(1,5,5)
    plt.scatter(X,Y,s=300,c=Fv_max,clip_on = False,vmin=0.7,vmax=1.2)
    #XY轴单位长度相同
    plt.gca().set_aspect(1)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['bottom'].set_visible(False)
    ax5.spines['left'].set_visible(False)
    plt.colorbar()
    plt.title('Fv_max',fontsize=20)

    plt.suptitle(runID.replace("_","-"))
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(save_path1,'offset_spatialinfo_'+runID+'.png'),dpi=500,bbox_inches = 'tight')
    plt.close()
    #plt.show()

    '''保存变量'''
    with open(os.path.join(save_path1,'Fv_max.txt'),'wb') as A1:
        pickle.dump(Fv_max,A1)
    with open(os.path.join(save_path1,'C.txt'),'wb') as A1:
        pickle.dump(C,A1)
    with open(os.path.join(save_path1,'shift_info_sml.txt'),'wb') as A1:
        pickle.dump(shift_info_sml,A1)
    with open(os.path.join(save_path1,'shift_info_wav.txt'),'wb') as A1:
        pickle.dump(shift_info_wav,A1)





def F_getFvShift(C_BF,C_FF):
    step=0.2
    indexLength=len(C_BF)
    FF_max_value=np.max(C_FF)
    FF_max_pos=np.argmax(C_FF)
    if FF_max_pos>0:
        BF_1=C_BF[0:FF_max_pos]
    else:
        BF_1 = C_BF[FF_max_pos]
    BF_2=C_BF[FF_max_pos:indexLength]
    BF_max_value1=np.max(BF_1)
    BF_max_pos1=np.argmax(BF_1)
    BF_max_value2=np.max(BF_2)
    BF_max_pos2=np.argmax(BF_2)
    '''python不用加1'''
    BF_max_pos2=BF_max_pos2+FF_max_pos
    fv_shift1=(FF_max_pos-BF_max_pos1)*step
    fv_shift2=(BF_max_pos2-FF_max_pos)*step
    '''这里需要加1'''
    WDI_shift=((indexLength+1)/2-(FF_max_pos+1))*step
    return fv_shift1,fv_shift2,WDI_shift


# runID = '123'
# result_path =r'D:\bost_ui\output\z_get_imge'
# save_path = r'D:\bost_ui\output\z_get_imge'
#
# Zscan_data_extract_singlechip(runID, result_path, save_path)















