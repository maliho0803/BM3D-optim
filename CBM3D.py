# -*- coding: utf-8 -*-

import cv2
import PSNR
import numpy as np
import multiprocessing as mp
import multiprocessing.pool as pool
from functools import partial


cv2.setUseOptimized(True)

# Parameters initialization
sigma = 25
Threshold_Hard3D = 2.7*sigma           # Threshold for Hard Thresholding
First_Match_threshold = 2500             # 用于计算block之间相似度的阈值
Step1_max_matched_cnt = 16              # 组最大匹配的块数
Step1_Blk_Size = 6                     # block_Size即块的大小，8*8
Step1_Blk_Step = 3                     # Rather than sliding by one pixel to every next reference block, use a step of Nstep pixels in both horizontal and vertical directions.
Step1_Search_Step = 3                   # 块的搜索step
Step1_Search_Window = 39                # Search for candidate matching blocks in a local neighborhood of restricted size NS*NS centered

Second_Match_threshold = 400           # 用于计算block之间相似度的阈值
Step2_max_matched_cnt = 16
Step2_Blk_Size = 6
Step2_Blk_Step = 3
Step2_Search_Step = 3
Step2_Search_Window = 39

Beta_Kaiser = 2.0


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
    if LX < 0:   LX = 0
    elif RX > _noisyImg.shape[0]:   LX = _noisyImg.shape[0]-_WindowSize
    if LY < 0:   LY = 0
    elif RY > _noisyImg.shape[0]:   LY = _noisyImg.shape[0]-_WindowSize

    return np.array((LX, LY), dtype=int)


def Step1_fast_match(_noisyImg, _BlockPoint):
    """快速匹配"""
    '''
    *返回邻域内寻找和当前_block相似度最高的几个block,返回的数组中包含本身
    *_noisyImg:噪声图像
    *_BlockPoint:当前block的坐标及大小
    '''
    (present_x, present_y) = _BlockPoint  # 当前坐标
    Blk_Size = Step1_Blk_Size
    Search_Step = Step1_Search_Step
    Threshold = First_Match_threshold
    max_matched = Step1_max_matched_cnt
    Window_size = Step1_Search_Window

    blk_positions = np.zeros((max_matched, 2), dtype=int)  # 用于记录相似blk的位置
    Final_similar_blocks = np.zeros((max_matched, Blk_Size, Blk_Size, 3), dtype=float)

    img = _noisyImg[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size, :]
    Final_similar_blocks[0, :, :, 0] = cv2.dct(img[:, :, 0].astype(np.float64))  # 对目标作block作二维余弦变换
    Final_similar_blocks[0, :, :, 1] = cv2.dct(img[:, :, 1].astype(np.float64))
    Final_similar_blocks[0, :, :, 2] = cv2.dct(img[:, :, 2].astype(np.float64))

    blk_positions[0, :] = _BlockPoint

    Window_location = Define_SearchWindow(np.squeeze(_noisyImg[:,:,0]), _BlockPoint, Window_size, Blk_Size)
    blk_num = (Window_size-Blk_Size)/Search_Step  # 确定最多可以找到多少相似blk
    blk_num = int(blk_num)
    (present_x, present_y) = Window_location

    similar_blocks = np.zeros((blk_num**2, Blk_Size, Blk_Size), dtype=float)
    similar_blocksU = np.zeros((blk_num ** 2, Blk_Size, Blk_Size), dtype=float)
    similar_blocksV = np.zeros((blk_num ** 2, Blk_Size, Blk_Size), dtype=float)
    m_Blkpositions = np.zeros((blk_num**2, 2), dtype=int)
    Distances = np.zeros(blk_num**2, dtype=float)  # 记录各个blk与它的相似度

    # 开始在_Search_Window中搜索,初始版本先采用遍历搜索策略,这里返回最相似的几块
    matched_cnt = 0
    for i in range(blk_num):
        for j in range(blk_num):
            tem_img = _noisyImg[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size, :]
            dct_Tem_img = cv2.dct(tem_img[:,:,0].astype(np.float64))
            m_Distance = np.linalg.norm((Final_similar_blocks[0, :, :, 0] - dct_Tem_img))**2 / (Blk_Size**2)

            # 下面记录数据自动不考虑自身(因为已经记录)
            if m_Distance < Threshold and m_Distance > 0:  # 说明找到了一块符合要求的
                similar_blocks[matched_cnt, :, :] = dct_Tem_img
                similar_blocksU[matched_cnt, :, :] = cv2.dct(_noisyImg[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size, 1].astype(np.float64))
                similar_blocksV[matched_cnt, :, :] = cv2.dct(_noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size, 2].astype(np.float64))
                m_Blkpositions[matched_cnt, :] = (present_x, present_y)
                Distances[matched_cnt] = m_Distance
                matched_cnt += 1
            present_y += Search_Step
        present_x += Search_Step
        present_y = Window_location[1]
    Distances = Distances[:matched_cnt]
    Sort = Distances.argsort()

    # 统计一下找到了多少相似的blk
    if matched_cnt < max_matched:
        Count = matched_cnt + 1
    else:
        Count = max_matched

    if Count > 0:
        for i in range(1, Count):
            Final_similar_blocks[i, :, :, 0] = similar_blocks[Sort[i-1], :, :]
            Final_similar_blocks[i, :, :, 1] = similar_blocksU[Sort[i - 1], :, :]
            Final_similar_blocks[i, :, :, 2] = similar_blocksV[Sort[i - 1], :, :]
            blk_positions[i, :] = m_Blkpositions[Sort[i-1], :]

    return Final_similar_blocks, blk_positions, Count


def Step1_3DFiltering(_similar_blocks):
    '''
    *3D变换及滤波处理
    *_similar_blocks:相似的一组block,这里已经是频域的表示
    *要将_similar_blocks第三维依次取出,然在频域用阈值滤波之后,再作反变换
    '''
    statis_nonzero = np.array([0,0,0])  # 非零元素个数
    m_Shape = _similar_blocks.shape

    for k in range(m_Shape[3]):
        for i in range(m_Shape[1]):
            for j in range(m_Shape[2]):
                tem_Vct_Trans = cv2.dct(_similar_blocks[:, i, j, k])
                tem_Vct_Trans[np.abs(tem_Vct_Trans[:]) < Threshold_Hard3D] = 0.
                statis_nonzero[k] += tem_Vct_Trans.nonzero()[0].size
                _similar_blocks[:, i, j, k] = np.ravel(cv2.idct(tem_Vct_Trans))

    return _similar_blocks, statis_nonzero


def Aggregation_hardthreshold(_similar_blocks, blk_positions, m_basic_img, m_wight_img, _nonzero_num, Count, Kaiser):
    '''
    *对3D变换及滤波后输出的stack进行加权累加,得到初步滤波的图片
    *_similar_blocks:相似的一组block,这里是频域的表示
    *对于最后的数组，乘以凯撒窗之后再输出
    '''
    for k in range(3):
        _shape = _similar_blocks.shape
        if _nonzero_num[k] < 1:
            _nonzero_num[k] = 1
        block_wight = (1./_nonzero_num[k]) * Kaiser
        for i in range(Count):
            point = blk_positions[i, :]
            tem_img = (1./_nonzero_num[k])*cv2.idct(_similar_blocks[i, :, :, k]) * Kaiser
            m_basic_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2], k] += tem_img
            m_wight_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2], k] += block_wight

def tread_function_s1(start_w, end_w, start_h, end_h, blk_step, block_Size, _noisyImg):
    Basic_img, m_Wight, m_Kaiser = init(_noisyImg, Step1_Blk_Size, Beta_Kaiser)
    (width, height, _) = _noisyImg.shape
    for i in range(start_w, end_w):
        for j in range(start_h, end_h):
            # m_blockPoint当前参考图像的顶点
            m_blockPoint = Locate_blk(i, j, blk_step, block_Size, width, height)       # 该函数用于保证当前的blk不超出图像范围
            Similar_Blks, Positions, Count = Step1_fast_match(_noisyImg, m_blockPoint)
            Similar_Blks, statis_nonzero = Step1_3DFiltering(Similar_Blks)
            Aggregation_hardthreshold(Similar_Blks, Positions, Basic_img, m_Wight, statis_nonzero, Count, m_Kaiser)

    return Basic_img, m_Wight

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

    # 开始逐block的处理,+2是为了避免边缘上不够
    # for i in range(int(Width_num+2)):
    #     for j in range(int(Height_num+2)):
    #         # m_blockPoint当前参考图像的顶点
    #         m_blockPoint = Locate_blk(i, j, blk_step, block_Size, width, height)       # 该函数用于保证当前的blk不超出图像范围
    #         Similar_Blks, Positions, Count = Step1_fast_match(_noisyImg, m_blockPoint)
    #         Similar_Blks, statis_nonzero = Step1_3DFiltering(Similar_Blks)
    #         Aggregation_hardthreshold(Similar_Blks, Positions, Basic_img, m_Wight, statis_nonzero, Count, m_Kaiser)

    threads = mp.cpu_count()
    print(threads)
    pool = mp.Pool(processes=threads)
    cal_func = partial(tread_function_s1, blk_step=blk_step, block_Size=block_Size, _noisyImg=_noisyImg)

    #将图像分为6个区域进行并行计算
    split_w = 4
    split_h = 2
    input_args = []

    for i in range(split_w):
        for j in range(split_h):
            temp = (int(i * Width_num/split_w), int((i + 1) * Width_num / split_w + 2), int(j * Height_num / split_h), int((j + 1) * Height_num / split_h + 2))
            input_args.append(temp)

    res = pool.starmap(cal_func, input_args)
    print(len(res))

    for k in range(split_h * split_w):
        Basic_img += res[k][0]
        m_Wight += res[k][1]

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
    (present_x, present_y) = _BlockPoint  # 当前坐标
    Blk_Size = Step2_Blk_Size
    Threshold = Second_Match_threshold
    Search_Step = Step2_Search_Step
    max_matched = Step2_max_matched_cnt
    Window_size = Step2_Search_Window

    blk_positions = np.zeros((max_matched, 2), dtype=int)  # 用于记录相似blk的位置
    Final_similar_blocks = np.zeros((max_matched, Blk_Size, Blk_Size, 3), dtype=float)
    Final_noisy_blocks = np.zeros((max_matched, Blk_Size, Blk_Size, 3), dtype=float)

    img = _Basic_img[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size, :]
    Final_similar_blocks[0, :, :, 0] = cv2.dct(img[:, :, 0].astype(np.float32))
    Final_similar_blocks[0, :, :, 1] = cv2.dct(img[:, :, 1].astype(np.float32))
    Final_similar_blocks[0, :, :, 2] = cv2.dct(img[:, :, 2].astype(np.float32))

    n_img = _noisyImg[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size, :]
    Final_noisy_blocks[0, :, :, 0] = cv2.dct(n_img[:, :, 0].astype(np.float32))
    Final_noisy_blocks[0, :, :, 1] = cv2.dct(n_img[:, :, 1].astype(np.float32))
    Final_noisy_blocks[0, :, :, 2] = cv2.dct(n_img[:, :, 2].astype(np.float32))

    blk_positions[0, :] = _BlockPoint

    Window_location = Define_SearchWindow(np.squeeze(_noisyImg[:,:,0]), _BlockPoint, Window_size, Blk_Size)
    blk_num = (Window_size-Blk_Size)/Search_Step  # 确定最多可以找到多少相似blk
    blk_num = int(blk_num)
    (present_x, present_y) = Window_location

    similar_blocks = np.zeros((blk_num**2, Blk_Size, Blk_Size), dtype=float)
    similar_blocksU = np.zeros((blk_num ** 2, Blk_Size, Blk_Size), dtype=float)
    similar_blocksV = np.zeros((blk_num ** 2, Blk_Size, Blk_Size), dtype=float)
    m_Blkpositions = np.zeros((blk_num**2, 2), dtype=int)
    Distances = np.zeros(blk_num**2, dtype=float)  # 记录各个blk与它的相似度

    # 开始在_Search_Window中搜索,初始版本先采用遍历搜索策略,这里返回最相似的几块
    matched_cnt = 0
    for i in range(blk_num):
        for j in range(blk_num):
            tem_img = _Basic_img[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size, 0]
            dct_Tem_img = cv2.dct(tem_img.astype(np.float32))
            m_Distance = np.linalg.norm((Final_similar_blocks[0, :, :, 0]-dct_Tem_img))**2 / (Blk_Size**2)

            # 下面记录数据自动不考虑自身(因为已经记录)
            if m_Distance < Threshold and m_Distance > 0:
                similar_blocks[matched_cnt, :, :] = dct_Tem_img
                similar_blocksU[matched_cnt, :, :] = cv2.dct(_Basic_img[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size, 1].astype(np.float64))
                similar_blocksV[matched_cnt, :, :] = cv2.dct(_Basic_img[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size, 2].astype(np.float64))
                m_Blkpositions[matched_cnt, :] = (present_x, present_y)
                Distances[matched_cnt] = m_Distance
                matched_cnt += 1
            present_y += Search_Step
        present_x += Search_Step
        present_y = Window_location[1]
    Distances = Distances[:matched_cnt]
    Sort = Distances.argsort()

    # 统计一下找到了多少相似的blk
    if matched_cnt < max_matched:
        Count = matched_cnt + 1
    else:
        Count = max_matched

    if Count > 0:
        for i in range(1, Count):
            Final_similar_blocks[i, :, :, 0] = similar_blocks[Sort[i-1], :, :]
            Final_similar_blocks[i, :, :, 1] = similar_blocksU[Sort[i - 1], :, :]
            Final_similar_blocks[i, :, :, 2] = similar_blocksV[Sort[i - 1], :, :]
            blk_positions[i, :] = m_Blkpositions[Sort[i-1], :]

            (present_x, present_y) = m_Blkpositions[Sort[i-1], :]
            n_img = _noisyImg[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size, :]
            for k in range(3):
                Final_noisy_blocks[i, :, :, k] = cv2.dct(n_img[:,:,k].astype(np.float64))

    return Final_similar_blocks, Final_noisy_blocks, blk_positions, Count


def Step2_3DFiltering(_Similar_Bscs, _Similar_Imgs):
    '''
    *3D维纳变换的协同滤波
    *_similar_blocks:相似的一组block,这里是频域的表示
    *要将_similar_blocks第三维依次取出,然后作dct,在频域进行维纳滤波之后,再作反变换
    *返回的Wiener_wight用于后面Aggregation
    '''
    m_Shape = _Similar_Bscs.shape
    Wiener_wight = np.zeros((m_Shape[1], m_Shape[2], 3), dtype=float)

    for k in range(m_Shape[3]):
        for i in range(m_Shape[1]):
            for j in range(m_Shape[2]):
                tem_vector = _Similar_Bscs[:, i, j, k]
                tem_Vct_Trans = np.matrix(cv2.dct(tem_vector))
                Norm_2 = np.float64(tem_Vct_Trans.T * tem_Vct_Trans)
                m_weight = Norm_2/(Norm_2 + sigma**2)
                if m_weight != 0:
                    Wiener_wight[i, j, k] = 1./(m_weight**2 * sigma**2)
                # else:
                #     Wiener_wight[i, j] = 10000
                tem_vector = _Similar_Imgs[:, i, j, k]
                tem_Vct_Trans = m_weight * cv2.dct(tem_vector)
                _Similar_Bscs[:, i, j, k] = np.ravel(cv2.idct(tem_Vct_Trans))

    return _Similar_Bscs, Wiener_wight


def Aggregation_Wiener(_Similar_Blks, _Wiener_wight, blk_positions, m_basic_img, m_wight_img, Count, Kaiser):
    '''
    *对3D变换及滤波后输出的stack进行加权累加,得到初步滤波的图片
    *_similar_blocks:相似的一组block,这里是频域的表示
    *对于最后的数组，乘以凯撒窗之后再输出
    '''
    _shape = _Similar_Blks.shape
    for k in range(3):
        block_wight = _Wiener_wight[:,:,k] # * Kaiser

        for i in range(Count):
            point = blk_positions[i, :]
            tem_img = _Wiener_wight[:,:,k] * cv2.idct(_Similar_Blks[i, :, :, k]) # * Kaiser
            m_basic_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2], k] += tem_img
            m_wight_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2], k] += block_wight

def tread_function_s2(start_w, end_w, start_h, end_h, blk_step, block_Size, _noisyImg, _basicImg):
    m_img, m_Wight, m_Kaiser = init(_noisyImg, block_Size, Beta_Kaiser)
    (width, height, _) = _noisyImg.shape
    for i in range(start_w, end_w):
        for j in range(start_h, end_h):
            m_blockPoint = Locate_blk(i, j, blk_step, block_Size, width, height)
            Similar_Blks, Similar_Imgs, Positions, Count = Step2_fast_match(_basicImg, _noisyImg, m_blockPoint)
            Similar_Blks, Wiener_wight = Step2_3DFiltering(Similar_Blks, Similar_Imgs)
            print(Similar_Blks.shape, Wiener_wight.shape)
            Aggregation_Wiener(Similar_Blks, Wiener_wight, Positions, m_img, m_Wight, Count, m_Kaiser)

    return m_img, m_Wight

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

    # for i in range(int(Width_num+2)):
    #     for j in range(int(Height_num+2)):
    #         m_blockPoint = Locate_blk(i, j, blk_step, block_Size, width, height)
    #         Similar_Blks, Similar_Imgs, Positions, Count = Step2_fast_match(_basicImg, _noisyImg, m_blockPoint)
    #         Similar_Blks, Wiener_wight = Step2_3DFiltering(Similar_Blks, Similar_Imgs)
    #         Aggregation_Wiener(Similar_Blks, Wiener_wight, Positions, m_img, m_Wight, Count, m_Kaiser)

    threads = mp.cpu_count()
    pool = mp.Pool(processes=threads)
    cal_func = partial(tread_function_s2, blk_step=blk_step, block_Size=block_Size, _noisyImg=_noisyImg, _basicImg= _basicImg)

    # 将图像分为6个区域进行并行计算
    split_w = 4
    split_h = 2
    input_args = []

    for i in range(split_w):
        for j in range(split_h):
            temp = (int(i * Width_num / split_w), int((i + 1) * Width_num / split_w + 2), int(j * Height_num / split_h),
                    int((j + 1) * Height_num / split_h + 2))
            input_args.append(temp)

    res = pool.starmap(cal_func, input_args)

    for k in range(split_h * split_w):
        m_img += res[k][0]
        m_Wight += res[k][1]

    m_img[:, :, :] /= m_Wight[:, :, :]
    Final = np.array(m_img, dtype=int).astype(np.uint8)
    return Final


if __name__ == '__main__':
    import glob
    img_paths = glob.glob('./*jpg')
    for img_name in img_paths:
        print(img_name)
        cv2.setUseOptimized(True)   # OpenCV 中的很多函数都被优化过（使用 SSE2，AVX 等）。也包含一些没有被优化的代码。使用函数 cv2.setUseOptimized() 来开启优化。
        # img_name = "./test.jpg"  # 图像的路径
        img = cv2.resize(cv2.imread(img_name), (256, 256))

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

        # cv2.imwrite('./' + img_name.split('/')[-1].replace('.jpg', '_blur.jpg'), Final_img)
        # cv2.imwrite('./' + img_name.split('/')[-1], img)

