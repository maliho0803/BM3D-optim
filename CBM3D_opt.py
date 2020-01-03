# -*- coding: utf-8 -*-

import cv2
# import PSNR
import numpy as np
# import pysnooper
import time
from numba import jit, prange
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from functools import partial
from scipy.fftpack import dct, idct
# import multithreading
cv2.setUseOptimized(True)

# Parameters initialization
Sigma = 25
Threshold_Hard3D = 2.7*Sigma           # Threshold for Hard Thresholding
First_Match_threshold = 2500             # 用于计算block之间相似度的阈值
Step1_max_matched_cnt = 8              # 组最大匹配的块数
Step1_Blk_Size = 4                     # block_Size即块的大小，8*8
Step1_Blk_Step = 2                      # Rather than sliding by one pixel to every next reference block, use a step of Nstep pixels in both horizontal and vertical directions.
Step1_Search_Step = 2                   # 块的搜索step
Step1_Search_Window = 19                # Search for candidate matching blocks in a local neighborhood of restricted size NS*NS centered

Second_Match_threshold = 400           # 用于计算block之间相似度的阈值
Step2_max_matched_cnt = 8
Step2_Blk_Size = 4
Step2_Blk_Step = 2
Step2_Search_Step = 2
Step2_Search_Window = 19

Beta_Kaiser = 2.0

DCT_BATCH = 100
THREAD_NUM = mp.cpu_count()

def init(img, _blk_size, _Beta_Kaiser):
    """该函数用于初始化，返回用于记录过滤后图像以及权重的数组,还有构造凯撒窗"""
    m_shape = img.shape
    m_img = np.zeros(m_shape, dtype=float)
    m_wight = np.zeros(m_shape, dtype=float)
    K = np.matrix(np.kaiser(_blk_size, _Beta_Kaiser))
    m_Kaiser = np.array(K.T * K)            # 构造一个凯撒窗
    return m_img, m_wight, m_Kaiser


def Locate_blk(i, j, blk_step, block_Size, width, height):
    '''该函数用于保证当前的blk不超出图像范围'''
    if i*blk_step+block_Size < width:
        point_x = i*blk_step
    else:
        point_x = width - block_Size

    if j*blk_step+block_Size < height:
        point_y = j*blk_step
    else:
        point_y = height - block_Size
    m_blockPoint = np.array((point_x, point_y), dtype=int)  # 当前参考图像的顶点
    return m_blockPoint


def Define_SearchWindow(_noisyImg, _BlockPoint, _WindowSize, Blk_Size):
    """该函数返回一个二元组（x,y）,用以界定_Search_Window顶点坐标"""
    point_x = _BlockPoint[0]  # 当前坐标
    point_y = _BlockPoint[1]  # 当前坐标

    # 获得SearchWindow四个顶点的坐标
    LX = point_x+Blk_Size/2-_WindowSize/2     # 左上x
    LY = point_y+Blk_Size/2-_WindowSize/2     # 左上y
    RX = LX+_WindowSize                       # 右下x
    RY = LY+_WindowSize                       # 右下y

    # 判断一下是否越界
    if LX < 0:   
        LX = 0
    elif RX > _noisyImg.shape[1]-Blk_Size:   
        LX = _noisyImg.shape[1]-_WindowSize-Blk_Size
    if LY < 0:   
        LY = 0
    elif RY > _noisyImg.shape[0]-Blk_Size:   
        LY = _noisyImg.shape[0]-_WindowSize-Blk_Size
    return np.array((LX, LY), dtype=int)


def batch_dct(images):
    dim = images.ndim
    if dim == 3: 
        return dct(dct(images, axis=0, norm="ortho"), axis=1, norm="ortho")
    elif dim == 4:
        return dct(dct(images, axis=1, norm="ortho"), axis=2, norm="ortho")
    else: 
        raise ValueError("input images must be 3d or 4d array")

def batch_idct(images):
    dim = images.ndim
    if dim == 3: 
        return idct(idct(images, axis=0, norm="ortho"), axis=1, norm="ortho")
    elif dim == 4:
        return idct(idct(images, axis=1, norm="ortho"), axis=2, norm="ortho")
    else: 
        raise ValueError("input images must be 3d or 4d array")

def Step1_fast_match(_noisyImg, _BlockPoint):
    """快速匹配"""
    '''
    *返回邻域内寻找和当前_block相似度最高的几个block,返回的数组中包含本身
    *_noisyImg:噪声图像
    *_BlockPoint:当前block的坐标及大小
    '''
    (init_x, init_y) = _BlockPoint  # 当前坐标
    Blk_Size = Step1_Blk_Size
    Search_Step = Step1_Search_Step
    Threshold = First_Match_threshold
    max_matched = Step1_max_matched_cnt
    Window_size = Step1_Search_Window
    img = _noisyImg[init_x: init_x+Blk_Size, init_y: init_y+Blk_Size, :]

    Window_location = Define_SearchWindow(np.squeeze(_noisyImg[:,:,0]), _BlockPoint, Window_size, Blk_Size)
    blk_num = (Window_size-Blk_Size)/Search_Step  # 确定最多可以找到多少相似blk
    blk_num = int(blk_num)
    (present_x, present_y) = Window_location

    blk_positions = np.zeros((blk_num**2, 2), dtype=int)  # 用于记录相似blk的位置
    Final_similar_blocks = np.zeros((blk_num**2, Blk_Size, Blk_Size, 3), dtype=float)
    Final_similar_blocks[:] = batch_dct(img)
    blk_positions[0, :] = _BlockPoint

    # 开始在_Search_Window中搜索,初始版本先采用遍历搜索策略,这里返回最相似的几块
    x_steps = np.arange(present_x, present_x+Search_Step*blk_num, Search_Step)
    y_steps = np.arange(present_y, present_y+Search_Step*blk_num, Search_Step)
    yv, xv = np.meshgrid(y_steps, x_steps)
    xv = np.ravel(xv)
    yv = np.ravel(yv)
    xy_grid = np.dstack([xv, yv]).reshape((-1, 2))
    image_batch = []
    for grid in xy_grid:
        x, y = grid
        tem_img = _noisyImg[x: x+Blk_Size, y: y+Blk_Size, :]
        image_batch.append(tem_img)

    dct_img_batch = batch_dct(np.array(image_batch))
    distances = np.linalg.norm((Final_similar_blocks[:, :, :, 0]-dct_img_batch[:, :, :, 0]), axis=(1, 2))**2 / (Blk_Size**2)
    match_bool = (distances > 0) & (distances < Threshold)
    m_count = np.sum(match_bool)

    m_distances = distances[match_bool]
    m_dct_img = dct_img_batch[match_bool]
    m_blk_position = xy_grid[match_bool]
    if m_count < max_matched:
        Count = m_count + 1
    else:
        Count = max_matched
    if Count > 1:
        max_sort_idx = np.argsort(m_distances)[:Count]
        Final_similar_blocks[1:len(max_sort_idx) + 1,:] = m_dct_img[max_sort_idx[:]]
        blk_positions[1:len(max_sort_idx) + 1 ,:] = m_blk_position[max_sort_idx[:]]

    return Final_similar_blocks[:Count], blk_positions[:Count], Count


def Step1_3DFiltering(_similar_blocks):
    '''
    *3D变换及滤波处理
    *_similar_blocks:相似的一组block,这里已经是频域的表示
    *要将_similar_blocks第三维依次取出,然在频域用阈值滤波之后,再作反变换
    '''
    uni_flag = 1
    if _similar_blocks.ndim == 4:
        axis = 0
    elif _similar_blocks.ndim == 5:
        axis = 1
    else:
        # print("Warning: different shape input")
        axis = 0
        uni_flag = 0
    # statis_nonzero = np.array([0,0,0])  # 非零元素个数
    def _3d_filter(x, axis): 
        vct_trans = dct(x, axis=axis, norm="ortho")
        vct_trans[np.abs(vct_trans) < Threshold_Hard3D] = 0.
        statis_nonzero = np.sum(vct_trans != 0, axis=(0+axis, 1+axis, 2+axis))
        vct_rev = idct(vct_trans, axis=axis, norm="ortho")
        return vct_rev, statis_nonzero

    if uni_flag:
        vct_rev, statis_nonzero = _3d_filter(_similar_blocks, axis)
        return vct_rev, statis_nonzero
    else:
        vct_revs, statis_nonzeros = [], []
        for simi_block in _similar_blocks:
            vct_rev, statis_nonzero = _3d_filter(simi_block, axis)
            vct_revs.append(vct_rev)
            statis_nonzeros.append(statis_nonzero)
        return vct_revs, statis_nonzeros


def Aggregation_hardthreshold(_similar_blocks, blk_positions, m_basic_img, m_wight_img, _nonzero_num, Count, Kaiser):
    '''
    *对3D变换及滤波后输出的stack进行加权累加,得到初步滤波的图片
    *_similar_blocks:相似的一组block,这里是频域的表示
    *对于最后的数组，乘以凯撒窗之后再输出
    '''
    block_size = _similar_blocks.shape
    _nonzero_num[_nonzero_num < 1] = 1
    block_w = np.concatenate([np.expand_dims(Kaiser/i, -1) for i in _nonzero_num], axis=-1)
    r_sim_blocks = batch_idct(_similar_blocks) * block_w
    cur_m_basic_img = m_basic_img.copy()
    cur_m_wight_img = m_wight_img.copy()
    for i in range(Count):
        point = blk_positions[i, :]
        cur_m_basic_img[point[0]:point[0]+block_size[1], point[1]:point[1]+block_size[2], :] += r_sim_blocks[i]
        cur_m_wight_img[point[0]:point[0]+block_size[1], point[1]:point[1]+block_size[2], :] += block_w
    return cur_m_basic_img, cur_m_wight_img

def _s1_thread_func(xy_list, Basic_img, noisyImg, m_Wight, image_w, image_h, m_Kaiser, blk_step, block_Size): 
    basic_img_cp = Basic_img.copy()
    m_wight_cp = m_Wight.copy()
    batch_size = DCT_BATCH
    total_count = len(xy_list)
    batch_similar_blks, batch_positions, batch_count = [], [], 0
    for idx, (i, j) in enumerate(xy_list):
        m_blockPoint = Locate_blk(i, j, blk_step, block_Size, image_w, image_h)       # 该函数用于保证当前的blk不超出图像范围
        Similar_Blks, Positions, Count = Step1_fast_match(noisyImg, m_blockPoint)

        batch_similar_blks.append(Similar_Blks)
        batch_positions.append(Positions)
        batch_count += Count
        if ((idx+1) % batch_size == 0) or (idx == total_count-1):
            merge_batch_similar_blks = np.array(batch_similar_blks)
            merge_batch_positions = np.concatenate(batch_positions, axis=0)
            Similar_Blks, statis_nonzero = Step1_3DFiltering(merge_batch_similar_blks)
            statis_nonzero = np.sum(statis_nonzero, axis=0)
            Similar_Blks = np.concatenate(Similar_Blks, axis=0)
            basic_img_cp, m_wight_cp = Aggregation_hardthreshold(Similar_Blks, merge_batch_positions, basic_img_cp, m_wight_cp, statis_nonzero, batch_count, m_Kaiser)
            batch_similar_blks, batch_positions, batch_count = [], [], 0
    return basic_img_cp, m_wight_cp

def BM3D_1st_step(_noisyImg):
    """第一步,基本去噪"""
    # 初始化一些参数：
    (width, height, _) = _noisyImg.shape   # 得到图像的长宽
    block_Size = Step1_Blk_Size         # 块大小
    blk_step = Step1_Blk_Step           # N块步长滑动
    Width_num = (width - block_Size)/blk_step
    Height_num = (height - block_Size)/blk_step

    # 初始化几个数组
    Basic_img, m_Wight, m_Kaiser = init(_noisyImg, Step1_Blk_Size, Beta_Kaiser)
    threads = THREAD_NUM
    Width_num = (width - block_Size)/blk_step
    Height_num = (height - block_Size)/blk_step
    xv, yv = np.meshgrid(np.arange(Width_num+2), np.arange(Height_num+2))
    xy_grid = list(zip(np.ravel(xv), np.ravel(yv)))
    pool = mp.Pool(processes=threads)
    cal_func = partial(_s1_thread_func, Basic_img=Basic_img, noisyImg=_noisyImg, m_Wight=m_Wight, 
                       image_w=width, image_h=height, blk_step=blk_step, block_Size=block_Size, m_Kaiser=m_Kaiser)
    # input_args = list(zip(basic_image_list, noisy_image_list, m_wight_list, image_w, image_h))
    # res = pool.map(cal_func, input_args)
    xy_list = np.array_split(xy_grid, threads)
    res = pool.map(cal_func, xy_list)

    # for i in xy_list:
        # res = cal_func(i)
    basic_image_list = [x[0] for x in res]
    w_wight_list = [x[1] for x in res]
    Basic_img = np.sum(basic_image_list, axis=0)
    m_Wight = np.sum(w_wight_list, axis=0)

    Basic_img[:, :, :] /= m_Wight[:, :, :]
    basic = np.array(Basic_img, dtype=int).astype(np.uint8)
    return basic


def Step2_fast_match(_Basic_img, _noisyImg, _BlockPoint):
    '''
    *快速匹配算法,返回邻域内寻找和当前_block相似度最高的几个block,要同时返回basicImg和IMG
    *_Basic_img: 基础去噪之后的图像
    *_noisyImg:噪声图像
    *_BlockPoint:当前block的坐标及大小
    '''
    (init_x, init_y) = _BlockPoint  # 当前坐标
    Blk_Size = Step2_Blk_Size
    Search_Step = Step2_Search_Step
    Threshold = Second_Match_threshold
    max_matched = Step2_max_matched_cnt
    Window_size = Step2_Search_Window

    Final_noisy_blocks = np.zeros((max_matched, Blk_Size, Blk_Size, 3), dtype=float)
    basic_img = _Basic_img[init_x:init_x+Blk_Size, init_y:init_y+Blk_Size, :]

    Window_location = Define_SearchWindow(np.squeeze(_noisyImg[:,:,0]), _BlockPoint, Window_size, Blk_Size)
    blk_num = (Window_size-Blk_Size)/Search_Step  # 确定最多可以找到多少相似blk
    blk_num = int(blk_num)
    blk_positions = np.zeros((blk_num**2, 2), dtype=int)  # 用于记录相似blk的位置
    Final_similar_blocks = np.zeros((blk_num**2, Blk_Size, Blk_Size, 3), dtype=float)
    Final_similar_blocks[:] = batch_dct(basic_img)
    blk_positions[0, :] = _BlockPoint
    (present_x, present_y) = Window_location

    # 开始在_Search_Window中搜索,初始版本先采用遍历搜索策略,这里返回最相似的几块
    x_steps = np.arange(present_x, present_x+Search_Step*blk_num, Search_Step)
    y_steps = np.arange(present_y, present_y+Search_Step*blk_num, Search_Step)
    yv, xv = np.meshgrid(y_steps, x_steps)
    xv = np.ravel(xv)
    yv = np.ravel(yv)
    xy_grid = np.dstack([xv, yv]).reshape((-1, 2))
    image_batch = []
    for grid in xy_grid:
        x, y = grid
        tem_img = _Basic_img[x: x+Blk_Size, y: y+Blk_Size, :]
        image_batch.append(tem_img)
    dct_img_batch = batch_dct(np.array(image_batch))
    distances = np.linalg.norm((Final_similar_blocks[:, :, :, 0]-dct_img_batch[:, :, :, 0]), axis=(1, 2))**2 / (Blk_Size**2)
    match_bool = (distances > 0) & (distances < Threshold)
    m_count = np.sum(match_bool)
    m_distances = distances[match_bool]
    m_dct_img = dct_img_batch[match_bool]
    m_blk_position = xy_grid[match_bool]
    if m_count < max_matched:
        Count = m_count + 1
    else:
        Count = max_matched
    if Count > 1:
        max_sort_idx = np.argsort(m_distances)[:Count]
        Final_similar_blocks[1:len(max_sort_idx) + 1,:] = m_dct_img[max_sort_idx[:]]
        blk_positions[1:len(max_sort_idx) + 1,:] = m_blk_position[max_sort_idx[:]]

    n_imgs = []
    for x, y in blk_positions[0:Count]:
        n_img = _noisyImg[x:x+Blk_Size, y: y+Blk_Size, :]
        n_imgs.append(n_img)
    Final_noisy_blocks[:Count] = batch_dct(np.array(n_imgs))
    return Final_similar_blocks[:Count], Final_noisy_blocks[:Count], blk_positions[:Count], Count

def Step2_3DFiltering(_Similar_Bscs, _Similar_Imgs):
    '''
    *3D维纳变换的协同滤波
    *_similar_blocks:相似的一组block,这里是频域的表示
    *要将_similar_blocks第三维依次取出,然后作dct,在频域进行维纳滤波之后,再作反变换
    *返回的Wiener_wight用于后面Aggregation
    '''

    uni_flag = 1
    if _Similar_Bscs.ndim == 4:
        axis = 0
    elif _Similar_Bscs.ndim == 5:
        axis = 1
    else:
        # print("Warning: different shape input")
        axis = 0
        uni_flag = 0

    def _wiener(x): 
        x_shape = x.shape
        a = np.matrix(x.T.reshape((-1, x_shape[0])))
        a = (a * a.T).diagonal()[0]
        a = np.array(a).reshape(x_shape[1:][::-1]).T
        m_weight = a / (a + Sigma**2)
        return m_weight

    def _3d_wiener_filter(simi_bscs, simi_imgs, axis): 
        x_shape = simi_bscs.shape
        vct_trans = dct(simi_bscs, axis=axis, norm="ortho")
        if axis == 0:
            m_weight = _wiener(vct_trans)
            m_weight = np.tile(np.expand_dims(m_weight, 0), (x_shape[0], 1, 1, 1))
            # print('zhoumi0: ', m_weight.shape)
        else: 
            m_weights = []
            for vct in vct_trans: 
                m_weight = _wiener(vct) 
                m_weight = np.tile(np.expand_dims(m_weight, 0), (x_shape[1],1,1,1))
                m_weights.append(m_weight)
            m_weight = np.array(m_weights)
            # print('zhoumi1: ', m_weight.shape)
        wiener_weight = np.where(m_weight!=0., 1./(m_weight**2*Sigma**2), m_weight)
        vct_trans = dct(simi_imgs, axis=axis, norm="ortho")
        vct_trans = m_weight * vct_trans
        simi_bscs = idct(vct_trans, axis=axis, norm="ortho")
        return simi_bscs, wiener_weight

    if uni_flag:
        simi_bscs, m_weight = _3d_wiener_filter(_Similar_Bscs, _Similar_Imgs, axis)
        # print('dfff',simi_bscs.shape, m_weight.shape)
        return simi_bscs, m_weight
    else:
        simi_bscs, m_weights = [], []
        for simi_block, simi_image in zip(_Similar_Bscs, _Similar_Imgs):
            # print(simi_block.shape, simi_image.shape)
            simi_bsc, m_weight = _3d_wiener_filter(simi_block, simi_image, axis)
            simi_bscs.append(simi_bsc)
            m_weights.append(m_weight)
        # print('adf', np.array(simi_bscs).shape, np.array(m_weights).shape)
        return simi_bscs, m_weights

def Aggregation_Wiener(_Similar_Blks, _Wiener_wight, blk_positions, m_basic_img, m_wight_img, Count):
    '''
    *对3D变换及滤波后输出的stack进行加权累加,得到初步滤波的图片
    *_similar_blocks:相似的一组block,这里是频域的表示
    *对于最后的数组，乘以凯撒窗之后再输出
    '''
    block_size = _Similar_Blks.shape

    r_sim_blocks = batch_idct(_Similar_Blks) * _Wiener_wight
    cur_m_basic_img = m_basic_img.copy()
    cur_m_wight_img = m_wight_img.copy()
    ndim = _Wiener_wight.ndim
    for i in range(Count):
        point = blk_positions[i, :]
        cur_m_basic_img[point[0]:point[0]+block_size[1], point[1]:point[1]+block_size[2], :] += r_sim_blocks[i]
        if ndim == 4:
            cur_m_wight_img[point[0]:point[0]+block_size[1], point[1]:point[1]+block_size[2], :] += _Wiener_wight[i]
        else:
            cur_m_wight_img[point[0]:point[0]+block_size[1], point[1]:point[1]+block_size[2], :] += _Wiener_wight
    return cur_m_basic_img, cur_m_wight_img

def _s2_thread_func(xy_list, Basic_img, noisyImg, m_img, m_Wight, image_w, image_h, m_Kaiser, blk_step, block_Size):
    basic_img_cp = Basic_img.copy()
    m_wight_cp = m_Wight.copy()
    m_img_cp = m_img.copy()
    m_wight_cp2 = m_Wight.copy()
    m_img_cp2 = m_img.copy()
    batch_size = DCT_BATCH
    total_count = len(xy_list)
    batch_similar_blks, batch_similar_imgs, batch_positions, batch_count = [], [], [], 0

    for idx, (i, j) in enumerate(xy_list):
        m_blockPoint = Locate_blk(i, j, blk_step, block_Size, image_w, image_h)       # 该函数用于保证当前的blk不超出图像范围
        Similar_Blks, Similar_Imgs, Positions, Count = Step2_fast_match(Basic_img, noisyImg, m_blockPoint)
        batch_similar_blks.append(Similar_Blks)
        batch_similar_imgs.append(Similar_Imgs)
        batch_positions.append(Positions)
        batch_count += Count
        if (((idx+1) % batch_size == 0) and (idx != 0)) or (idx == total_count-1):
            merge_batch_similar_blks = np.array(batch_similar_blks)
            merge_batch_similar_imgs = np.array(batch_similar_imgs)
            merge_batch_positions = np.concatenate(batch_positions, axis=0)
            Similar_Blks, wiener_w = Step2_3DFiltering(merge_batch_similar_blks, merge_batch_similar_imgs)
            Similar_Blks = np.concatenate(Similar_Blks, axis=0)
            wiener_w = np.concatenate(wiener_w, axis=0)
            # print('zhoumi', Similar_Blks.shape, wiener_w.shape)
            m_img_cp, m_wight_cp = Aggregation_Wiener(Similar_Blks, wiener_w, merge_batch_positions, m_img_cp, m_wight_cp, batch_count)
            batch_similar_blks, batch_positions, batch_similar_imgs, batch_count = [], [], [], 0
    return m_img_cp, m_wight_cp

def BM3D_2nd_step(_basicImg, _noisyImg):
    '''Step 2. 最终的估计: 利用基本的估计，进行改进了的分组以及协同维纳滤波'''
    # 初始化一些参数：
    (width, height, _) = _noisyImg.shape
    block_Size = Step2_Blk_Size
    blk_step = Step2_Blk_Step
    Width_num = (width - block_Size)/blk_step
    Height_num = (height - block_Size)/blk_step

    # 初始化几个数组
    m_img, m_Wight, m_Kaiser = init(_noisyImg, block_Size, Beta_Kaiser)

    threads = THREAD_NUM
    xv, yv = np.meshgrid(np.arange(Width_num+2), np.arange(Height_num+2))
    xy_grid = list(zip(np.ravel(xv), np.ravel(yv)))
    pool = mp.Pool(processes=threads)
    _basicImg = _basicImg.astype(np.float64)
    cal_func = partial(_s2_thread_func, Basic_img=_basicImg, noisyImg=_noisyImg, m_img=m_img, m_Wight=m_Wight, 
                       image_w=width, image_h=height, blk_step=blk_step, block_Size=block_Size, m_Kaiser=m_Kaiser)
    xy_list = np.array_split(xy_grid, threads)
    res = pool.map(cal_func, xy_list)
    # res = cal_func(xy_grid)
    basic_image_list = [x[0] for x in res]
    w_wight_list = [x[1] for x in res]
    Basic_img = np.sum(basic_image_list, axis=0)
    m_Wight = np.sum(w_wight_list, axis=0)

    # print(Basic_img[-1], m_Wight[-1])
    Basic_img[:,:,:] /= (m_Wight[:,:,:])
    Final = np.array(Basic_img, dtype=int).astype(np.uint8)

    return Final


if __name__ == '__main__':
    import glob
    # img_paths = glob.glob('./*jpg')
    img_paths = glob.glob('./*jpg')
    for img_name in img_paths:
        print(img_name)
        cv2.setUseOptimized(True)   # OpenCV 中的很多函数都被优化过（使用 SSE2，AVX 等）。也包含一些没有被优化的代码。使用函数 cv2.setUseOptimized() 来开启优化。
        # img_name = "./test.jpg"  # 图像的路径
        img = cv2.resize(cv2.imread(img_name), (112, 112))

        e1 = cv2.getTickCount()
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        print(new_img.shape, new_img.dtype)

        Basic_img = cv2.cvtColor(BM3D_1st_step(new_img), cv2.COLOR_YUV2BGR)
        e2 = cv2.getTickCount()
        time = (e2 - e1) / cv2.getTickFrequency()
        print("The Processing time of the first step is %f s" % time)

        Final_img = cv2.cvtColor(BM3D_2nd_step(Basic_img, new_img), cv2.COLOR_YUV2BGR)
        e3 = cv2.getTickCount()
        time = (e3 - e2) / cv2.getTickFrequency()
        print ("The Processing time of the second step is %f s" % time)

        cv2.imshow('adffa', np.hstack([img, Basic_img, Final_img]))
        cv2.waitKey()

        # cv2.imwrite('new.jpg', Final_img)