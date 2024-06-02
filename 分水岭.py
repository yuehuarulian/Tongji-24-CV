from scipy import ndimage as ndi
from skimage import morphology,color,data,filters,segmentation,io
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans # 聚类算法
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
matplotlib.rc('font', family='Microsoft YaHei')
gtfile = './data/gt/'
def fsl(img):
    image =color.rgb2gray(img)
    image = image.astype(np.uint8)
    denoised = filters.rank.median(image, morphology.disk(2)) #过滤噪声
    #将梯度值低于10的作为开始标记点
    markers = filters.rank.gradient(denoised, morphology.disk(5)) <5
    markers = ndi.label(markers)[0]
    gradient = filters.rank.gradient(denoised, morphology.disk(2)) #计算梯度
    labels =segmentation.watershed(gradient, markers, mask=image) #基于梯度的分水岭算法
    labels = (labels*255./labels.max()).astype(np.uint8)
    binary_image = np.where(labels > 29, 0, 255).astype(np.uint8)
    return labels
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    # axes = axes.ravel()
    # ax0, ax1, ax2, ax3 = axes
    # ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    # ax0.set_title("Original")
    # ax1.imshow(gradient, cmap=plt.cm.gray, interpolation='nearest')
    # ax1.set_title("Gradient")
    # ax2.imshow(markers, cmap=plt.cm.gray, interpolation='nearest')
    # ax2.set_title("Markers")
    # ax3.imshow(binary_image, cmap=plt.cm.gray, interpolation='nearest')
    # ax3.set_title("Segmented")
    
    # for ax in axes:
    #     ax.axis('off')
    
    # fig.tight_layout()
    # plt.show()


def kmeans(image, color_space='RGB',pos = False):
    if color_space=='HSV':
        img = color.rgb2hsv(image)
    else:
        img = image
    h, w, c = img.shape  # (316, 474, 3) height width 通道
    print('height:' + str(h) + ',  width:' + str(w) + ',  channel:' + str(c))
    # 构建原始像素矩阵
    H = img[:,:,0]
    S = img[:,:,1]
    V = img[:,:,2]
    # 像素拼接
    # print(R)
    list_H = H.reshape([H.shape[0]*H.shape[1]])
    list_S = S.reshape([S.shape[0]*S.shape[1]])
    list_V = V.reshape([V.shape[0]*V.shape[1]])
    # print(list_R)
    print('像素拼接完成,一个通道的长度为' + str(len(list_H)))
    
    # 构建原始数据矩阵A   (3, 316*474)

    A = np.zeros(shape=(c+1,len(list_H)))
    A[0,:] = list_H
    A[1,:] = list_S
    A[2,:] = list_V
    f = fsl(image)
    A[3,:] = f.reshape(f.shape[0]*f.shape[1])
    # print(A.shape,A)
    k = 2
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, tol=0.001) # {'k-means++', 'random', ndarray}或者callable
    kmeans.fit(A.T) # k-means按列聚类所以转置一下
    labelIDs = np.unique(kmeans.labels_)
    label = kmeans.labels_ # 聚类标签
    core = kmeans.cluster_centers_.T # 聚类中心
    print('k-means聚类完成')
    # 构建数据矩阵B
    B = np.zeros(shape=A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[i,j] = label[j] * 255
    
    # 合成新的图像
    img_new = B[0,:].reshape(h,w)
    print('新的像素矩阵合成完毕')
    return img_new

# 导入图片
image_path = []
all_images = []
images = os.listdir('./data/images')

for image_name in images:
    if image_name.endswith('.jpg'):
        image_path.append('./data/images/' + image_name)
IOU_RGB = []
IOU_HSV = []
print (image_path)

for path in image_path:
    image = np.array(Image.open(path))
    # 高斯滤波
    import cv2
    image = cv2.GaussianBlur(image, (5, 5), 2)
    # 图像增强
    image = image * 1.3 - 50
    image[image > 255] = 255
    image[image < 0] = 0
    
    img_hsv = kmeans(image,'HSV')
    img_rgb = kmeans(image,'RGB')
    # 图片输出
    plt.subplot(1,3,1) # 一共1*2，放在第一个位置
    plt.xlabel('原始图像',fontsize=18)
    plt.imshow(image/255)
    
    plt.subplot(1,3,2)
    plt.xlabel('RGB分割后图像',fontsize=18)
    img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
    plt.imshow(img_rgb,cmap='gray')

    plt.subplot(1,3,3)
    plt.xlabel('HSV分割后图像',fontsize=18)
    img_hsv = np.clip(img_hsv, 0, 255).astype(np.uint8)
    plt.imshow(img_hsv,cmap='gray')

    plt.savefig('./data/增强RGB_HSV/' + os.path.basename(path))
    plt.show()

    # 检测IOU
    imggt = np.array(Image.open(gtfile + os.path.splitext(os.path.basename(path))[0] + '.png'))
    # 将图像转换为二进制形式
    img_binary = (img_rgb > 0).astype(np.uint8)  # 转换为二进制形式，非黑即白
    imggt_binary = (imggt > 0).astype(np.uint8)
    # 计算交集（两个二进制图像相交的部分）
    intersection = np.logical_and(img_binary, imggt_binary)
    # 计算并集（两个二进制图像的并集）
    union = np.logical_or(img_binary, imggt_binary)
    # 计算 IoU 值
    iou = np.sum(intersection) / np.sum(union)
    if iou<0.5:
        iou = 1-iou
    IOU_RGB.append(iou)


    # 检测IOU
    # 将图像转换为二进制形式
    img_binary = (img_hsv > 0).astype(np.uint8)  # 转换为二进制形式，非黑即白
    imggt_binary = (imggt > 0).astype(np.uint8)
    # 计算交集（两个二进制图像相交的部分）
    intersection = np.logical_and(img_binary, imggt_binary)
    # 计算并集（两个二进制图像的并集）
    union = np.logical_or(img_binary, imggt_binary)
    # 计算 IoU 值
    iou = np.sum(intersection) / np.sum(union)
    if iou<0.5:
        iou = 1-iou
    IOU_HSV.append(iou)

print(IOU_RGB)
print(f"平均IoU为:", sum(IOU_RGB)/len(IOU_RGB))
print(IOU_HSV)
print(f"平均IoU为:", sum(IOU_HSV)/len(IOU_HSV))

















