from copy import copy, deepcopy  # 导入深拷贝函数
import math  # 导入数学函数库
from cv2 import CHAIN_APPROX_NONE, RETR_LIST, imshow, merge, findContours, waitKey  # 导入OpenCV相关函数
from skimage import morphology,color,data,filters,segmentation,io  # 导入图像处理库
import numpy as np  # 导入数值计算库
from tqdm import trange  # 导入进度条显示函数
from tqdm import tqdm  # 导入进度条显示函数
from sklearn.cluster import KMeans  # 导入KMeans聚类算法
import colorsys
from PIL import Image
import os
from scipy import ndimage as ndi
import cv2
import matplotlib.pyplot as plt
imagefile = 'data/images/'
resultfile = 'data/slic_kmeans/'
gtfile = 'data/gt/'
class Cluster(object):  # 定义聚类类
    cluster_index = 1  # 类变量，用于分配聚类编号

    def __init__(self, h, w, l=0, a=0, b=0):  # 初始化方法
        self.update(h, w, l, a, b)  # 更新聚类信息
        self.pixels = []  # 该聚类包含的像素列表
        self.no = self.cluster_index  # 聚类编号
        self.label = 0  # 聚类标签，用于后续KMeans聚类
        Cluster.cluster_index += 1  # 更新聚类编号

    def update(self, h, w, l, a, b):  # 更新聚类信息方法
        self.h = h  # 聚类中心的高度
        self.w = w  # 聚类中心的宽度
        self.l = l  # 聚类中心的L通道值
        self.a = a  # 聚类中心的A通道值
        self.b = b  # 聚类中心的B通道值

    def __str__(self):  # 返回对象字符串表示方法
        return "{},{}:{} {} {} ".format(self.h, self.w, self.l, self.a, self.b)

    def __repr__(self):  # 返回对象字符串表示方法
        return self.__str__()

class SLICProcessor(object):  # 定义SLIC处理器类
    @staticmethod
    def open_image(path):  # 静态方法，用于打开图像
        """
        Return:
            3D array, row col [LAB]
        """
        rgb = io.imread(path)  # 读取图像
        if path[-3:] == 'png':  # 如果图像格式为png
            lab_arr = color.rgb2lab(rgb[:, :, 0:3])  # 转换为LAB色彩空间
        else:
            lab_arr = color.rgb2lab(rgb)  # 转换为LAB色彩空间
        return lab_arr  # 返回LAB图像数组

    @staticmethod
    def save_lab_image(path, lab_arr):  # 静态方法，用于保存LAB图像
        """
        Convert the array to RBG, then save the image
        :param path:
        :param lab_arr:
        :return:
        """
        rgb_arr = color.lab2rgb(lab_arr)  # 转换为RGB色彩空间
        rgb_arr_uint8 = (rgb_arr * 255).astype(np.uint8)  # 转换为8位整数类型
        io.imsave(path, rgb_arr_uint8)  # 保存图像

    def make_cluster(self, h, w):  # 创建聚类对象方法
        h = int(h)  # 将高度转换为整数类型
        w = int(w)  # 将宽度转换为整数类型
        return Cluster(h, w,  # 返回新的聚类对象
                       self.data[h][w][0],
                       self.data[h][w][1],
                       self.data[h][w][2])

    def __init__(self, filename, K, M, count):  # 初始化方法
        self.K = K  # 设定聚类数量
        self.M = M  # 设定超参数M
        self.count = count
        self.data = self.open_image(filename)  # 打开图像并转换为LAB色彩空间
        self.image_height = self.data.shape[0]  # 获取图像高度
        self.image_width = self.data.shape[1]  # 获取图像宽度
        self.N = self.image_height * self.image_width  # 获取像素总数
        self.S = int(math.sqrt(self.N / self.K))  # 计算超像素大小S

        self.clusters = []  # 存储聚类的列表
        self.label = {}  # 存储像素对应聚类的字典
        self.dis = np.full((self.image_height, self.image_width), np.inf)  # 初始化距离矩阵为无穷大

    def init_clusters(self):  # 初始化聚类方法
        h = self.S // 2  # 计算初始高度
        w = self.S // 2  # 计算初始宽度
        while h < self.image_height:  # 循环直到高度超过图像高度
            while w < self.image_width:  # 循环直到宽度超过图像宽度
                self.clusters.append(self.make_cluster(h, w))  # 创建聚类对象并添加到列表
                w += self.S  # 更新宽度
            w = self.S // 2  # 重置宽度
            h += self.S  # 更新高度

    def get_gradient(self, h, w):  # 获取梯度方法
        if w + 1 >= self.image_width:  # 如果宽度超出图像边界
            w = self.image_width - 2  # 设置为图像宽度-2
        if h + 1 >= self.image_height:  # 如果高度超出图像边界
            h = self.image_height - 2  # 设置为图像高度-2
        gradient = self.data[h + 1][w + 1][0] - self.data[h][w][0] + \
                   self.data[h + 1][w + 1][1] - self.data[h][w][1] + \
                   self.data[h + 1][w + 1][2] - self.data[h][w][2]
        return gradient  # 返回梯度值

    def move_clusters(self):  # 移动聚类中心方法
        for cluster in self.clusters:  # 遍历所有
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    new_gradient = self.get_gradient(_h, _w)
                    if new_gradient < cluster_gradient:
                        cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
                        cluster_gradient = new_gradient

    def assignment(self):
        for cluster in tqdm(self.clusters):
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):
                if h < 0 or h >= self.image_height: continue
                for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
                    if w < 0 or w >= self.image_width: continue
                    L, A, B = self.data[h][w]
                    Dc = math.sqrt(
                        math.pow(L - cluster.l, 2) +
                        math.pow(A - cluster.a, 2) +
                        math.pow(B - cluster.b, 2))
                    Ds = math.sqrt(
                        math.pow(h - cluster.h, 2) +
                        math.pow(w - cluster.w, 2))
                    D = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2))
                    if D < self.dis[h][w]:
                        if (h, w) not in self.label:
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[h][w] = D
        a = 1

    def update_cluster(self):
        for cluster in self.clusters:
            sum_h = sum_w = number = 0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                number += 1
            _h = int(sum_h / number)
            _w = int(sum_w / number)
            cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])

    def save_current_image(self, name):
        image_arr = np.copy(self.data)
        # imshow("r", image_arr)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            image_arr[cluster.h][cluster.w][0] = 0
            image_arr[cluster.h][cluster.w][1] = 0
            image_arr[cluster.h][cluster.w][2] = 0
        self.save_lab_image(resultfile + f"{self.count}_1.png", image_arr)
        mask_r = np.ones((self.image_height, self.image_width), np.uint8)*0
        for cluster in self.clusters:
            mask = np.ones((self.image_height, self.image_width), np.uint8)*0
            for x in cluster.pixels:
                mask[x[0], x[1]] = 255
            contours, _ = findContours(mask, RETR_LIST, CHAIN_APPROX_NONE)
            for contour in contours:
                for i in contour:
                    mask_r[i[0][1], i[0][0]] = 255
        for x in range(self.image_height):
            for y in range(self.image_width):
                if mask_r[x, y] == 255:
                    image_arr[x, y] = [100, 0, 0]
        self.save_lab_image(resultfile + f"{self.count}_2.png", image_arr)

    def iterate_10times(self):
        self.init_clusters()
        self.move_clusters()
        # for i in trange(10):
        self.assignment()
        self.update_cluster()
        # name = 'lenna_M{m}_K{k}_loop{loop}.png'.format(loop=0, m=self.M, k=self.K)
        self.save_current_image("123")

    def generate_result(self, K, img_d):
        clusters = deepcopy(self.clusters)
        temp_img = [[clusters[x].l, clusters[x].a, clusters[x].b, clusters[x].h, clusters[x].w, img_d[x]] for x in range(len(clusters))]
        # 将lab颜色空间的值转换为RGB颜色空间的值
        for j in range(len(temp_img)):
            # temp_img[j][0:3] = color.lab2rgb([temp_img[j][0:3]])[0]
            rgb = color.lab2rgb([temp_img[j][0:3]])[0]
            r, g, b = rgb[0], rgb[1], rgb[2]
            temp_img[j][0], temp_img[j][1], temp_img[j][2] = colorsys.rgb_to_hsv(r, g, b)
        # 对每一列进行归一化
        k = [1,1,1,0.3,0.3,1]
        for i in range(len(temp_img[0]) - 1):  # 遍历每一列
            column_values = [row[i] for row in temp_img]  # 获取当前列的所有值
            min_val = min(column_values)  # 计算当前列的最小值
            max_val = max(column_values)  # 计算当前列的最大值
            for j in range(len(temp_img)):# 对当前列的每个元素进行归一化
                temp_img[j][i] = (temp_img[j][i] - min_val) / (max_val - min_val) * k[i]
        # print(temp_img)
        kmeans = KMeans(n_clusters=K, init='k-means++', n_init=10, tol=0.001, random_state=3).fit(temp_img)
        for i in range(len(self.clusters)):
            self.clusters[i].label = kmeans.labels_[i]
        print(kmeans.cluster_centers_)
        mask = np.ones((self.image_height, self.image_width))
        img = merge([mask, mask, mask])
        for cluster in self.clusters:
            for pixel in cluster.pixels:
                # img[pixel[0], pixel[1]] = kmeans.cluster_centers_[cluster.label][0:3]
                if cluster.label == 0:
                    img[pixel[0], pixel[1]] = [100, 0, 0] # 黑色
                else:
                    img[pixel[0], pixel[1]] = [0, 0, 0] # 白色

        for num in range(K):
            mask = np.ones((self.image_height, self.image_width), np.uint8)*0
            for cluster in self.clusters:
                if cluster.label == num:
                    for pixel in cluster.pixels:
                        mask[pixel[0], pixel[1]] = 255
            contours, _ = findContours(mask, RETR_LIST, CHAIN_APPROX_NONE)
            for contour in contours:
                # if len(contour)<200: continue#可以注释掉这行获取小区域的分割
                for i in contour:
                   self.data[i[0][1], i[0][0]] = [100, 0, 0]
        
        self.save_lab_image(resultfile + f"{self.count}.png", img)
        self.save_lab_image(resultfile + f"{self.count}_4.png", self.data)




def fsl(img):
    image =color.rgb2gray(img)
    image = image.astype(np.uint8)
    denoised = filters.rank.median(image, morphology.disk(2)) #过滤噪声
    #将梯度值低于10的作为开始标记点
    markers = filters.rank.gradient(denoised, morphology.disk(5)) <5
    markers = ndi.label(markers)[0]
    gradient = filters.rank.gradient(denoised, morphology.disk(2)) #计算梯度
    labels =segmentation.watershed(gradient, markers, mask=image) #基于梯度的分水岭算法
    return labels

if __name__ == '__main__':
    # 图像灰度化
    IOU = []
    for i in range(1,17):
        img = np.array(Image.open(imagefile+f'/{i}.jpg'))
        f = fsl(img)
        f = f.reshape(f.shape[0]*f.shape[1])
        p = SLICProcessor(imagefile+f'/{i}.jpg', 100, 40, i)
        p.iterate_10times()
        p.generate_result(2,f)
        waitKey()
        

        # 检测IOU
        img = np.array(Image.open(resultfile + f"{i}.png").convert('L'))
        imggt = np.array(Image.open(gtfile + f"{i}.png"))
        # 将图像转换为二进制形式
        img_binary = (img > 0).astype(np.uint8)  # 转换为二进制形式，非黑即白
        imggt_binary = (imggt > 0).astype(np.uint8)
        # 计算交集（两个二进制图像相交的部分）
        intersection = np.logical_and(img_binary, imggt_binary)
        # 计算并集（两个二进制图像的并集）
        union = np.logical_or(img_binary, imggt_binary)
        # 计算 IoU 值
        iou = np.sum(intersection) / np.sum(union)
        if iou<0.5:
            iou = 1-iou
        IOU.append(iou)
        print(f"第{i}张图片的IoU为:", iou)
    print(IOU)
    print(f"平均IoU为:", sum(IOU)/len(IOU))
     