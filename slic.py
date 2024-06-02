import math
import cv2
from skimage import io, color
import numpy as np
from tqdm import trange
import os
 
gray_level=16
 
def feature_computer(p):#GLCM的特征提取
    # con:对比度反应了图像的清晰度和纹理的沟纹深浅。纹理越清晰反差越大对比度也就越大。
    # eng:熵（Entropy, ENT）度量了图像包含信息量的随机性，表现了图像的复杂程度。当共生矩阵中所有值均相等或者像素值表现出最大的随机性时，熵最大。
    # agm:角二阶矩（能量），图像灰度分布均匀程度和纹理粗细的度量。当图像纹理均一规则时，能量值较大；反之灰度共生矩阵的元素值相近，能量值较小。
    # idm:反差分矩阵又称逆方差，反映了纹理的清晰程度和规则程度，纹理清晰、规律性较强、易于描述的，值较大。
    Con = 0.0
    Eng = 0.0
    Asm = 0.0
    Idm = 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con += (i - j) * (i - j) * p[i][j]
            Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            if p[i][j] > 0.0:
                Eng += p[i][j] * math.log(p[i][j])
    return Asm, Con, -Eng, Idm
 
 
class Cluster(object):
    cluster_index = 1
    def __init__(self, h, w, l=0, a=0, b=0): #初始化
        self.update(h, w, l, a, b)
        self.pixels = []
        self.no = self.cluster_index
        Cluster.cluster_index += 1
 
    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b
 
    def __str__(self):
        return "{},{}:{} {} {} ".format(self.h, self.w, self.l, self.a, self.b)
 
    def __repr__(self):
        return self.__str__()
 
 
class SLICProcessor(object):
    @staticmethod
    def open_image(path):#将rgb图片转为lab图片
        rgb = io.imread(path)
        lab_arr = color.rgb2lab(rgb)
        return lab_arr
 
    @staticmethod
    def save_lab_image(path, lab_arr):
        rgb_arr = color.lab2rgb(lab_arr)
        rgb_arr_uint8 = (rgb_arr * 255).astype(np.uint8)
        # print(rgb_arr.dtype)
        io.imsave(path, rgb_arr_uint8)
 
    def make_cluster(self, h, w):
        h = int(h)
        w = int(w)
        return Cluster(h, w,
                       self.data[h][w][0],
                       self.data[h][w][1],
                       self.data[h][w][2])
 
    def __init__(self, filename, K, M,n):
        self.K = K
        self.M = M
        self.n = n   #第几张图片
        self.data = self.open_image(filename)
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))
 
        self.clusters = []
        self.label = {}
        self.dis = np.full((self.image_height, self.image_width), np.inf)
 
    def init_clusters(self):
        h = self.S / 2
        w = self.S / 2
        while h < self.image_height:
            while w < self.image_width:
                self.clusters.append(self.make_cluster(h, w))
                w += self.S
            w = self.S / 2
            h += self.S
 
    def get_gradient(self, h, w):
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2
 
        gradient = self.data[h + 1][w + 1][0] - self.data[h][w][0] + \
                   self.data[h + 1][w + 1][1] - self.data[h][w][1] + \
                   self.data[h + 1][w + 1][2] - self.data[h][w][2]
        return gradient
 
    def move_clusters(self):#确定聚类中心点
        for cluster in self.clusters:
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
        for cluster in self.clusters:
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
 
    def update_cluster(self):
        for cluster in self.clusters:
            sum_h = sum_w = number =0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                number += 1
            _h = int(sum_h / number)
            _w = int(sum_w / number)
            cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
 
    def getGlcm(self,d_y,d_x):#灰度共生矩阵
        #将一整张图片转化为灰度图  
        glcm_img = cv2.imread(f"./images/{self.n}.jpg", 0)
        for cluster in self.clusters:  #每个超像素块
            max_gray_level = number=0
            ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
            for p in cluster.pixels:
                number += 1
                if glcm_img[p[0]][p[1]] > max_gray_level:
                    max_gray_level = glcm_img[p[0]][p[1]]
            max_gray_level = max_gray_level + 1         #得到此超像素块中最大灰度级
            for pix in cluster.pixels:
                if max_gray_level > gray_level:         #若是最大灰度级大于设定灰度级，则将其调整为设定灰度级大小
                    glcm_img[pix[0]][pix[1]] = glcm_img[pix[0]][pix[1]] * gray_level / max_gray_level
 
            for pixe in cluster.pixels:            #再次逐像素遍历
                h_1 = pixe[0] + d_y                #设定边界
                w_1 = pixe[1] + d_x
                if h_1 > glcm_img.shape[0] - 1:
                    h_1 = glcm_img.shape[0] - 1
                if w_1 > glcm_img.shape[1] - 1:
                    w_1 = glcm_img.shape[1] - 1
                rows = glcm_img[pixe[0]][pixe[1]]
                cols = glcm_img[h_1][w_1]
                if rows >= 16:
                    rows = 15
                if cols >= 16:
                    cols = 15
                ret[rows][cols] += 1.0
            for i in range(gray_level):
                for j in range(gray_level):
                    ret[i][j] /= float(number)     #得到灰度生成矩阵
            asm, con, eng, idm = feature_computer(ret)
            string=str([asm, con, eng, idm])
            # 确保目录存在
            os.makedirs(f'data/{self.n}', exist_ok=True)
            with open(f'data/{self.n}/benign{self.n}_d_y={d_y}_d_x={d_x}.txt', 'a+') as f:
                f.write('\n'+ string)# C:/Users/Administrator/Desktop/SRP/Dataset_BUSI_with_GT/benign/benign ({i}).png
 
    def average(self):#数据平均值
        for cluster in self.clusters: #每一个超像素
            l = a = b = number =0
            for p in cluster.pixels:  #超像素中的每一个像素
                l += self.data[p[0]][p[1]][0]
                a += self.data[p[0]][p[1]][1]
                b += self.data[p[0]][p[1]][2]
                number += 1            #像素总个数
            strin=str([l/number,a/number,b/number]) #均值
            with open(f'data/{self.n}/benign{self.n}_average.txt', 'a+') as f: #保存数据
                f.write('\n' + strin)
 
    def fangcha(self):#数据方差
        for cluster in self.clusters:
            l = a = b = []
            for p in cluster.pixels:
                l.append(self.data[p[0]][p[1]][0])
                a.append(self.data[p[0]][p[1]][1])
                b.append(self.data[p[0]][p[1]][2])
            stri=str([np.var(l),np.var(a),np.var(b)]) #方差
            with open(f'data/{self.n}/benign{self.n}_variance.txt', 'a+') as f:
                f.write('\n' + stri)
 
    def save_current_image(self, name):
        image_arr = np.copy(self.data)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            image_arr[cluster.h][cluster.w][0] = 0
            image_arr[cluster.h][cluster.w][1] = 0
            image_arr[cluster.h][cluster.w][2] = 0
        self.save_lab_image(name, image_arr)
 
    def iterate_10times(self):
        self.init_clusters()
        self.move_clusters()
        for i in trange(10):
            self.assignment()
            self.update_cluster()
            if i == 9:
                self.getGlcm(0,1)#参数为d_y,d_x
                self.getGlcm(-1,0)
                self.getGlcm(1,0)
                self.getGlcm(1,1)
                self.getGlcm(1,-1)
                self.getGlcm(-1,1)
                self.getGlcm(-1,-1)
                self.average()#平均值
                self.fangcha()#方差
                name = 'benign{n}_lenna_M{m}_K{k}_loop{loop}.png'.format(n=self.n,loop=i, m=self.M, k=self.K)
                self.save_current_image(name)
 
if __name__ == '__main__':
    for i in range(1,7):
        p = SLICProcessor(f'./images/{i}.jpg', 200, 30,i)
        p.iterate_10times()
 