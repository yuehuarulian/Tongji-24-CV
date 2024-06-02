import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
from skimage import io, color  # 导入图像处理库
import matplotlib
matplotlib.rc('font', family='Microsoft YaHei')
class KMeans:
    def __init__(self, n_clusters:int, n_init = 10, tol = 0.001):
        self.n_clusters = n_clusters
        self.max_iter = n_init
        self.tol = tol
    
    def fit(self, X):
        # 随机初始化聚类中心
        self.centers = X[np.random.choice(range(len(X)), size=self.n_clusters, replace=False)]
        # print(self.centers)
        for _ in range(self.max_iter):
            # 计算每个样本到各个聚类中心的距离
            distances = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
            # 找出每个样本最近的聚类中心
            labels = np.argmin(distances, axis=1)
            
            # 更新聚类中心为每个簇的均值
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            
            # 如果聚类中心的变化小于阈值，停止迭代
            if np.allclose(new_centers, self.centers):
                break
            
            self.centers = new_centers
        
        self.labels_ = labels
        # self.inertia_ = sum(np.min(np.linalg.norm(X - self.centers[labels[i]], axis=1)) for i in range(len(X)))
    
    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        return np.argmin(distances, axis=1)
    
if __name__ == '__main__':
    img = np.array(Image.open("data/images/1.jpg"))
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

    A = np.zeros(shape=(c,len(list_H)))
    A[0,:] = list_H
    A[1,:] = list_S
    A[2,:] = list_V
    # print(A.shape,A)
    k = 2
    kmeans = KMeans(n_clusters=k,n_init=10, tol=0.001) # {'k-means++', 'random', ndarray}或者callable
    kmeans.fit(A.T) # k-means按列聚类所以转置一下
    labelIDs = np.unique(kmeans.labels_)
    label = kmeans.labels_ # 聚类标签
    # print(label)
    # core = kmeans.cluster_centers_.T # 聚类中心
    print('k-means聚类完成')
    # 构建数据矩阵B
    B = np.zeros(shape=A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[i,j] = label[j] * 255
            # B[i,j] = core[i,label[j]]
    # print(B)
    
    # 合成新的图像
    H_new = B[0,:].reshape(h,w)
    S_new = B[1,:].reshape(h,w)
    V_new = B[2,:].reshape(h,w)
    # print(R_new)
    # img_new = np.zeros(shape=(h,w,c))
    # img_new[:,:,0] = H_new
    # img_new[:,:,1] = S_new
    # img_new[:,:,2] = V_new
    img_new = np.zeros(shape=(h,w))
    img_new = H_new
    print('新的像素矩阵合成完毕')

    # 图片输出
    plt.subplot(1,2,1) # 一共1*2，放在第一个位置
    plt.xlabel('原始图像',fontsize=18)
    plt.imshow(img/255)
    
    plt.subplot(1,2,2)
    plt.xlabel('RGB分割后图像',fontsize=18)
    img_gray = np.clip(img_new, 0, 255).astype(np.uint8)
    plt.imshow(img_gray,cmap='gray')
    plt.show()
