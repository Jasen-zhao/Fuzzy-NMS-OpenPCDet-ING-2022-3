# -*- coding: utf-8 -*-
# @Time    : 2021/9/25 15:11
# @Author  : ZFC
# @File    : volume_cal.py
# @Task    :
import numpy as np


def volume_cal(dxyz):#dxyz是框的长宽高
    # 获取体积
    dx = dxyz[:, 0]  # 长
    dy = dxyz[:, 1]  # 宽
    dz = dxyz[:, 2]  # 高
    volume = dx*dy*dz  # 每个框的体积,.cpu().numpy()：转成numpy数据

    #归一化,(使其不严格为0，不严格为1)
    #fac = 0.99 /max(volume)
    #volume= volume* fac

    #np.savetxt('volume.csv', volume, delimiter=',')  # 将数据保存成csv，保存在服务器...OpenPCDet/tools

    return volume