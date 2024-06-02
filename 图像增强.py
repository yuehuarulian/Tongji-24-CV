import numpy as np
import matplotlib.pylab as plt
from PIL import Image
def make_transform_matrix1(image_arr, d0, ftype):
    '''
    构建理想高/低通滤波器
    INPUT  -> 图像数组, 通带半径, 类型
    '''
    transfor_matrix = np.zeros(image_arr.shape, dtype=np.float32)  # 构建滤波器
    w, h = transfor_matrix.shape
    for i in range(w):
        for j in range(h):
            distance = np.sqrt((i - w/2)**2 + (j - h/2)**2)
            if distance < d0:
                transfor_matrix[i, j] = 1
            else:
                transfor_matrix[i, j] = 0
    if ftype == 'low':
        return transfor_matrix
    elif ftype == 'high':
        return 1 - transfor_matrix


def make_transform_matrix2(image_arr, d0, ftype='low'):
    '''
    构建高斯高/低通滤波
    INPUT  -> 图像数组, 通带半径, 类型
    '''
    transfor_matrix = np.zeros(image_arr.shape, dtype=np.float32)  # 构建滤波器
    w, h = image_arr.shape
    for i in range(w):
        for j in range(h):
            distance = np.sqrt((i - w/2)**2 + (j - h/2)**2)
            transfor_matrix[i, j] = np.e ** (-1 * (distance ** 2 / (2 * d0 ** 2)))  # Gaussian滤波函数
    if ftype == 'low':
        return transfor_matrix
    elif ftype == 'high':
        return 1 - transfor_matrix


for i in range(1,17):
    # 图像灰度化
    img_arr = np.array(Image.open('data/images/'+str(i)+'.jpg').convert('L'))

    # 将图像从空间域转换到频率域
    f = np.fft.fft2(img_arr)
    fshift = np.fft.fftshift(f)
     
    # F_filter1 = make_transform_matrix1(img_arr, 30, 'low')# 生成低通滤波器
    # result1 = fshift*F_filter1# 滤波
    # img_d1 = np.abs(np.fft.ifft2(np.fft.ifftshift(result1)))# 将图像从频率域转换到空间域
    # plt.subplot(1,2,1);plt.xlabel('1',fontsize=18);plt.imshow(img_d1)
    
    F_filter2 = make_transform_matrix2(img_arr, 30, 'high')# 生成高通滤波器
    result2 = fshift*F_filter2# 滤波
    img_d2 = np.abs(np.fft.ifft2(np.fft.ifftshift(result2)))# 将图像从频率域转换到空间域
    img_d2[img_d2>8] = 255
    img_d2[img_d2<=8] = 0
    
    # plt.subplot(1,2,2)
    plt.xlabel('2',fontsize=18);plt.imshow(img_d2,cmap='binary')
    plt.show()

        