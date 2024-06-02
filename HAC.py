import cv2
import numpy as np
# from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import os

class AgglomerativeClustering:
    def __init__(self,n_clusters:int, linkage='single'): # ward 单链接（single linkage）、完全链接（complete linkage）和平均链接（average linkage）
        self.n_clusters = n_clusters
        self.linkage = linkage
    def fit(self, X:list):
        X = np.array(X)
        n_samples = X.shape[0]
        self.labels_ = np.arange(n_samples)
        indexs = np.arange(n_samples) # 记录ij对应的原来位置
        
        # 计算初始距离矩阵
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                distances[i, j] = np.linalg.norm(X[i] - X[j])
                distances[j, i] = distances[i, j]
        
        # 开始合并簇
        for _ in range(n_samples - self.n_clusters):
            # 找出最近的两个簇
            distances = np.nan_to_num(distances, nan=0)
            non_diagonal_distances = distances + np.eye(distances.shape[0]) * np.max(distances)
            i, j = np.unravel_index(np.argmin(non_diagonal_distances), non_diagonal_distances.shape)
            label_i = indexs[i]
            label_j = indexs[j]
            
            # 根据链接方式更新距离矩阵
            if self.linkage == 'single':
                distances[i] = np.minimum(distances[i], distances[j])
            elif self.linkage == 'complete':
                distances[i] = np.maximum(distances[i], distances[j])
            elif self.linkage == 'average':
                label = np.arange(distances.shape[0])
                mask = (label == i) | (label == j)
                distances[i] = np.mean(distances[mask], axis=0)
            elif self.linkage == 'ward':
                mask_i = self.labels_ == label_i
                mask_j = self.labels_ == label_j
                centroid_i = X[mask_i].mean(axis=0)
                centroid_j = X[mask_j].mean(axis=0)
                n_i = np.sum(mask_i)
                n_j = np.sum(mask_j)
                n_ij = n_i + n_j
                for k in range(distances.shape[1]):
                    if k==i or k==j:
                        continue
                    mask_k = self.labels_ == indexs[k]
                    d_ik = np.mean(np.linalg.norm(X[mask_k] - centroid_i, axis=1) ** 2)
                    d_jk = np.mean(np.linalg.norm(X[mask_k] - centroid_j, axis=1) ** 2)
                    # d_ik = np.linalg.norm(centroid_k - centroid_i) ** 2
                    # d_jk = np.linalg.norm(centroid_k - centroid_j) ** 2
                    d_ij = np.linalg.norm(centroid_i - centroid_j) ** 2
                    distances_i = np.sqrt((n_i * (d_ik + d_ij) - 2 * d_ij) / n_ij)
                    distances_j = np.sqrt((n_j * (d_jk + d_ij) - 2 * d_ij) / n_ij)
                    distances[i,k] = max(distances_i, distances_j)
            else:
                raise ValueError("Unsupported linkage type.")
            
            # 更新簇标签
            # indexs[indexs == j] = i
            self.labels_[self.labels_ == label_j] = label_i

            distances[:, i] = distances[i]
            distances = np.delete(distances, j, axis=0)
            distances = np.delete(distances, j, axis=1)
            indexs = np.delete(indexs, j, axis=0)
        
        # self.labels_ = np.unique(self.labels_)
    
    def predict(self, X):
        distances = np.zeros((X.shape[0], len(self.labels_)))
        for i, label in enumerate(self.labels_):
            cluster_points = X[self.labels_ == label]
            distances[:, i] = np.min(np.linalg.norm(X[:, np.newaxis] - cluster_points, axis=2), axis=1)
        return np.argmin(distances, axis=1)



if __name__ =='__main__':
    # 导入图片
    image_path = []
    all_images = []
    images = os.listdir('./data/images')
    
    for image_name in images:
        if image_name.endswith('.jpg'):
            image_path.append('./data/images/' + image_name)
    for path in image_path:
        # 读取图像
        img = cv2.imread(path)
        # 新的宽度和高度
        new_width = 100
        new_height = 120
        # 调整图像大小
        image = cv2.resize(img, (new_width, new_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 将图像数据转换为二维数组
        pixels = np.reshape(image, (-1, 3))
        print(pixels.shape)
        # 使用层次聚类进行图像分割
        n_clusters = 2  # 指定聚类的数量
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
        clustering.fit(pixels.astype(np.float32))
        
        # 获取聚类标签并将其重新形状为图像大小
        labels = np.reshape(clustering.labels_, image.shape[:2])
        
        # 创建每个聚类的颜色映射
        colors = [np.mean(image[labels == i], axis=(0, 1)) for i in range(n_clusters)]
        
        # 根据聚类标签将像素着色
        segmented_image = np.zeros_like(image)
        for i in range(n_clusters):
            segmented_image[labels == i] = colors[i]
        
        # 显示分割后的图像
        plt.imshow(segmented_image)
        plt.axis('off')
        plt.show()